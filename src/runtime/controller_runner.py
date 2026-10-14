"""Runtime orchestration for the online-learning MPC controller."""

import copy
import os
import platform
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from class_MPC import MPC
from utils.DataManagement import SaveAggregatedData
from utils.DERTF_StatesDefinition import main_status_readings
from utils.OPCUA.DERTF_SetPointsWriter import OPCUA_SetPoint
from project_paths import fmu_file, LOGS_DIR

if platform.system() == "Windows":
    from pyfmi import load_fmu


class ControllerRunner:
    """Own controller state and coordinate debug or live execution.

    The runner stores time-indexed plant measurements, predictions, controls,
    solver status, and costs, then dispatches each control step to ``MPC``.
    Debug mode uses the FMU path; live mode uses the plant status reader and
    OPC UA setpoint writer.
    """

    def __init__(self, params, debug=False):
        if not debug:
            raise RuntimeError(
                "Live mode is retained only to show the implementation boundary "
                "and is not enabled for public operation."
            )
        self.params = params
        self.opts = {"Debug": debug}
        self.controller = MPC(params)
        self.controller.Opts = self.opts
        self.controller.Param = params
        self.controller.total_time = params["tot_hours"] * 3600
        self.controller.time_step = params["Ts"]
        self.controller.num_steps = self.controller.total_time // self.controller.time_step + 1
        self.controller.t_step = 0
        self.controller.y_dict = None
        initialize_buffers(self.controller, params)
        self.controller.InitialDate = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_exception(self, message):
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file = os.path.join(LOGS_DIR, "Exceptions.log")
        with open(log_file, "a") as log:
            log.write(f"{datetime.now()}: {message}\n")

    def read_status(self):
        """Read the latest state from the plant or the debug fixture."""
        controller = self.controller
        if self.opts["Debug"]:
            delivery = [71.06958, 70.96986, 71.104965, 71.12856]
            return_temp = [60.12344, 60.224037, 60.187595, 60.183292]
            flow = [2.7702346, 2.7731984, 2.7713318, 2.7692516]
            if controller.t_step < controller.Param["T_C0"]:
                index = min(controller.t_step, len(delivery) - 1)
                controller.y_dict = {
                    "T_delivery": delivery[index],
                    "T_return": return_temp[index],
                    "flow": flow[index],
                }
            else:
                controller.y_dict = {
                    "T_delivery": controller.y_out[0, controller.t_step],
                    "T_return": controller.y_out[1, controller.t_step],
                    "flow": controller.y_out[2, controller.t_step],
                }
            return

        try:
            controller.y_dict = main_status_readings()
        except Exception as error:
            self._report_exception("ReadStatus", error, "Keeping previous state values.")

    def set_points_to_plant(self, solved=True):
        """Write computed setpoints to the plant using OPC UA."""
        controller = self.controller
        try:
            index = controller.t_step - 1
            get_value = controller.opti.value if solved else controller.opti.debug.value
            data_map = controller.SetPointsWriter.DataMap
            data_map["Power_HL1"]["Value"] = -1e-3 * get_value(controller.u_out[0, index])
            data_map["Power_HL2"]["Value"] = -1e-3 * get_value(controller.u_out[1, index])
            data_map["Power_HL3"]["Value"] = -1e-3 * get_value(controller.u_out[2, index])
            data_map["Power_HL4"]["Value"] = -1e-3 * get_value(controller.u_out[3, index])
            data_map["T_out_GB"]["Value"] = get_value(controller.u_out[4, index])
            data_map["T_out_EB"]["Value"] = get_value(controller.u_out[5, index])

            for key in ("Power_HL1", "Power_HL2", "Power_HL3", "Power_HL4"):
                if abs(data_map[key]["Value"]) < 10:
                    data_map[key]["Value"] = 32
            for key in ("T_out_GB", "T_out_EB"):
                if data_map[key]["Value"] < 45:
                    data_map[key]["Value"] = 72
        except Exception as error:
            self._report_exception("SetPointsToDERTF", error)

        node_data = list(controller.SetPointsWriter.DataMap.values())
        if controller.SetPointsWriter.client is None:
            controller.SetPointsWriter.connect()

        for attempt in range(5):
            try:
                controller.SetPointsWriter.write_node(node_data)
                break
            except Exception as error:
                if attempt == 4:
                    self._report_exception("OPC UA write", error)
                else:
                    try:
                        controller.SetPointsWriter.disconnect()
                    except Exception:
                        pass
                    controller.SetPointsWriter.connect()
                    time.sleep(1)

    def set_points_to_fmu(self):
        """Send the generated control action to the FMU simulation."""
        controller = self.controller
        index = controller.t_step - 1
        time_step = controller.time_step
        time_points = np.arange(0, time_step * index + 1, time_step)
        input_array = np.column_stack([
            time_points,
            controller.u_out[0, : index + 1],
            controller.u_out[1, : index + 1],
            controller.u_out[2, : index + 1],
            controller.u_out[3, : index + 1],
            controller.u_out[4, : index + 1],
            controller.u_out[5, : index + 1],
        ])

        controller.fmu.set(["Ts_L4", "T_return", "mf_supply"], controller.y_out[:, 0])
        result = controller.fmu.simulate(
            start_time=time_step * controller.Param["T_C0"],
            final_time=time_step * (index + 1),
            options=controller.fmu_opts,
            input=(
                ["PowerLoad1", "PowerLoad2", "PowerLoad3", "PowerLoad4", "Tref_GB", "Tref_EB"],
                input_array,
            ),
        )
        controller.y_out[0, index + 1] = result["Ts_L4"][-1]
        controller.y_out[1, index + 1] = result["T_return"][-1]
        controller.y_out[2, index + 1] = result["mf_supply"][-1]
        controller.fmu.reset()

    def solve_step(self):
        """Run one read, optimize, persist, and apply cycle."""
        controller = self.controller
        index = copy.deepcopy(controller.t_step)
        self.read_status()
        controller.y_out[0, index] = controller.y_dict["T_delivery"]
        controller.y_out[1, index] = controller.y_dict["T_return"]
        controller.y_out[2, index] = controller.y_dict["flow"]

        try:
            if controller.Param["Model"] == "BNNExp" and index > controller.Param["T_C0"] - 1:
                controller.theta_current = controller.bayesian_correction(
                    index, controller.y_out, controller.u_out
                )
                controller.theta_history[index, :, :] = controller.theta_current
        except Exception as error:
            self._report_exception("Bayesian correction", error)

        try:
            flag = 1
            slack_tolerance = 1e-3
            result = controller.mpc_controller(
                index, controller.Param["T_C0"], controller.y_out, controller.u_out,
                controller.y_rnn_out, flag,
            )
            (
                controller.u_control, controller.slack, controller.y_nnarx,
                controller.computation_time[index], controller.sigma_current,
                controller.slack_explo, controller.sol_feas[index],
            ) = result
            if np.any(controller.slack_explo <= slack_tolerance):
                print("exploration ON")
            else:
                flag = 0
                print(" exploration OFF")
                result = controller.mpc_controller(
                    index, controller.Param["T_C0"], controller.y_out, controller.u_out,
                    controller.y_rnn_out, flag,
                )
                (
                    controller.u_control, controller.slack, controller.y_nnarx,
                    controller.computation_time[index], controller.sigma_current,
                    controller.slack_explo, controller.sol_feas[index],
                ) = result
            controller.exploration[0, index] = flag
            sigma = np.asarray(controller.sigma_current)
            if sigma.ndim == 0:
                sigma = np.full((1, controller.Param["n_out"]), sigma)
            elif sigma.ndim == 1:
                sigma = sigma.reshape(1, -1)
            controller.sigma[:, index] = sigma[0, :].reshape(controller.Param["n_out"])
        except Exception as error:
            self._report_exception("MPC controller", error)
            controller.y_nnarx = controller.y_prec

        controller.u_out[:, index] = controller.u_control
        if index < controller.Param["T_C0"] - 1:
            controller.s_out[:, index] = controller.slack[:, 0]
        else:
            controller.s_out[:, index] = (
                controller.slack[:, 0]
                if controller.Param["Model"] == "BNNExp"
                else controller.slack
            )
        controller.y_rnn_out[index + 1, :] = controller.y_nnarx[:, 0]
        controller.y_pred_out[:, index + 1] = controller.y_prec[:, 0]
        controller.Pb_pred_out[0, index] = controller.Pb_prec[0, 0]
        controller.Pb_pred_out[1, index] = controller.Pb_prec[1, 0]

        gamma_pred_gas = 0.034
        cop = 0.8
        efficiency_eb = 0.8
        row = int(np.floor(index / 3))
        electricity_cost = controller.ensure2D(controller.c_el)
        gamma_pred_el = electricity_cost[row, 0] / 1000
        time_step = 300
        controller.final_cost[index] = (
            gamma_pred_gas * (1 / 1000 * time_step / 3600 * controller.Pb_pred_out[0, index] / cop)
            + gamma_pred_el * (1 / 1000 * time_step / 3600 * controller.Pb_pred_out[1, index] / efficiency_eb)
        )
        controller.t_step += 1

        try:
            SaveAggregatedData(controller, index)
            if np.mod(index, controller.Param["T_C0"] - 1) == 0 or index == 0:
                if index > 0 and not self.opts["Debug"]:
                    self.set_points_to_plant()
            if index >= controller.Param["T_C0"] and self.opts["Debug"]:
                self.set_points_to_fmu()
        except Exception as error:
            self._report_exception("saving data or sending setpoints", error)

    def run(self):
        """Initialize the selected execution mode and run it."""
        controller = self.controller
        if controller.Param["Model"] == "BNNExp":
            controller.y_rnn_out[0, :] = controller.y_prec[:, 0]
            controller.y_pred_out[:, 0] = controller.y_prec[:, 0]

        if self.opts["Debug"]:
            self._configure_fmu()
            for _ in range(0, controller.num_steps - 1):
                self.solve_step()
            return

        self._configure_live_plant()
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            self.read_status, "interval", seconds=10,
            next_run_time=datetime.now() + timedelta(seconds=2),
        )
        scheduler.add_job(
            self.solve_step, "interval", seconds=controller.time_step,
            next_run_time=datetime.now() + timedelta(seconds=5),
        )
        scheduler.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.shutdown(wait=False)

    def _configure_live_plant(self):
        self.controller.SetPointsWriter = OPCUA_SetPoint()
        self.controller.SetPointsWriter.DataMap = {
            "Power_HL1": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_001", "Value": None},
            "Power_HL2": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_002", "Value": None},
            "Power_HL3": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_003", "Value": None},
            "Power_HL4": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_004", "Value": None},
            "T_out_GB": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_005", "Value": None},
            "T_out_EB": {"NodeID": "ns=0;s=PUBLIC.SYNTHETIC.CONTROL_006", "Value": None},
        }

    def _configure_fmu(self):
        controller = self.controller
        controller.fmu = load_fmu(fmu_file("DHN_MC_V3_FMU_EB_60.fmu"))
        fmu_options = controller.fmu.simulate_options()
        fmu_options["CVode_options"]["rtol"] = 1e-4
        fmu_options["CVode_options"]["atol"] = 1e-4
        controller.fmu_opts = fmu_options
        init_steps = min(5, controller.num_steps, controller.N)
        controller.y_out[:, :init_steps] = controller.y_prec[:, :init_steps]
        controller.u_out[:, :init_steps] = controller.u_prec[:, :init_steps]
        controller.y_pred_out[:, 0] = controller.y_prec[:, 0]
        controller.y_rnn_out[0, :] = controller.y_prec[:, 0]

    def _report_exception(self, location, error, prefix=None):
        message = f"Error occurred in {location}: {error}"
        if prefix:
            print(prefix)
        trace = traceback.extract_tb(error.__traceback__)
        if trace:
            last_trace = trace[-1]
            message += f"\nFile: {last_trace.filename}\nLine: {last_trace.lineno}"
        self.log_exception(message)
        print(message)


def initialize_buffers(controller, params):
    """Allocate time-indexed arrays used to store the control-loop results.

    The main state and control histories have shapes
    ``(n_out, num_steps)`` and ``(n_inp, num_steps)`` respectively. Prediction
    histories use the same output ordering as the controller, while
    ``theta_history`` has shape ``(num_steps, n_x, n_out)``. BNNExp stores its
    output-sized slack history; the other strategies store ``n_slack`` slack
    values per step.
    """
    controller.theta_history = np.zeros([controller.num_steps, params["n_x"], params["n_out"]])
    controller.sigma = np.zeros([params["n_out"], controller.num_steps])
    controller.exploration = np.zeros((1, controller.num_steps))
    controller.u_out = np.zeros([params["n_inp"], controller.num_steps])
    controller.y_out = np.zeros([params["n_out"], controller.num_steps])
    controller.y_rnn_out = np.zeros([controller.num_steps, params["n_out"]])
    controller.y_pred_out = np.zeros([params["n_out"], controller.num_steps])
    controller.Pb_pred_out = np.zeros([2, controller.num_steps])
    controller.s_out = (
        np.zeros([params["n_out"], controller.num_steps])
        if params["Model"] == "BNNExp"
        else np.zeros([params["n_slack"], controller.num_steps])
    )
    controller.final_cost = np.zeros([controller.num_steps])
    controller.computation_time = np.zeros([controller.num_steps])
    controller.sol_feas = np.zeros([controller.num_steps])
    controller.time_points = np.arange(
        0, controller.total_time + 1, controller.time_step
    )[:controller.num_steps]
