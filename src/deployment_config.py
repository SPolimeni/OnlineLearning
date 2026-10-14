"""Environment-backed live deployment settings.

The defaults remain the existing non-operational placeholders. Deployment
owners must provide private values through environment variables before live
use; importing this module never creates a client or opens a connection.
"""

import os


OPCUA_SERVER_URL = os.environ.get(
    "ONLINELEARNING_OPCUA_SERVER_URL",
    "",
)
INFLUXDB_URL = os.environ.get("ONLINELEARNING_INFLUXDB_URL", "")
INFLUXDB_TOKEN = os.environ.get("ONLINELEARNING_INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.environ.get("ONLINELEARNING_INFLUXDB_ORG", "")
INFLUXDB_BUCKET = os.environ.get("ONLINELEARNING_INFLUXDB_BUCKET", "")