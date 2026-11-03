"""Adapter exports; numerical implementation is audited parallel_flow_runtime."""
import runtime_acceleration as base
import parallel_flow_runtime as flow
frozen_hash=base.frozen_hash
full_frozen_hash_audit=base.full_frozen_hash_audit
configure=flow.configure
shutdown=flow.shutdown
loss_function=flow.loss_function
def process_info():
 service=base._prefetch.flow_service
 return dict(pid=service.process.pid,physical_GPU=service.ready['physical_GPU'],frozen_RAFT_hash=service.ready['model_hash'])
