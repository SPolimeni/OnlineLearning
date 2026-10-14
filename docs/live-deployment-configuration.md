# Live Deployment Configuration

Live mode is deployment-specific and is not exercised by the offline tests.
The connectors read private settings from environment variables when they are
constructed:

| Setting | Environment variable |
| --- | --- |
| OPC UA server URL | `ONLINELEARNING_OPCUA_SERVER_URL` |
| InfluxDB URL | `ONLINELEARNING_INFLUXDB_URL` |
| InfluxDB token | `ONLINELEARNING_INFLUXDB_TOKEN` |
| InfluxDB organization | `ONLINELEARNING_INFLUXDB_ORG` |
| InfluxDB bucket | `ONLINELEARNING_INFLUXDB_BUCKET` |

The source defaults remain non-operational placeholders for compatibility. Do
not add credentials, certificates, internal endpoints, or plant node maps to
source control or publication artifacts. Deployment owners must provide and
approve those values privately before selecting live mode.

The bundled InfluxDB configuration files contain only small synthetic metadata
fixtures for offline inspection. Real deployment mappings, node identifiers,
plant tags, zones, topology, and measurement registries must remain private
and must be supplied through an owner-approved deployment process.