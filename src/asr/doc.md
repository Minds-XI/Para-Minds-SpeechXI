# command to update grpc
```bash
python -m grpc_tools.protoc -I ../../protos --python_out=. \
         --grpc_python_out=. ../../protos/asr.proto

```
then update the line of code in `asr_pb2_grpc.py` to include the subpackage name 
```python 
from asr.grpc_generated import asr_pb2 as asr__pb2
```