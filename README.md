# cass-ceisdpt-prototype
A prototypic python implementation for a service selection algorithm that considers the effects of context interdependencies and possible service interruptions through external influences. 

### Docker image usage
Run the docker image with `docker run -it rikuxan/cass-ceisdpt-prototype:latest`  
Include your own service definition file by mounting it in the script folder: `docker run -it -v /path/to/your/file:/opt/problem-data.json rikuxan/cass-ceisdpt-prototype:latest`  
Command line parameters can be supplied by simply adding them after the docker command (e.g. `docker run -it rikuxan/cass-ceisdpt-prototype:latest --print-detailed-compositions`