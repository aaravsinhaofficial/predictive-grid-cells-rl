import os,sys,json,runpy
from pathlib import Path
checkout=Path(sys.argv[1]).resolve();original=Path(sys.argv[2]).resolve();script=sys.argv[3];record=Path(sys.argv[4]);extra=sys.argv[5:];reads=set();writes=set();missing=set();blocked=[]
def hook(event,args):
 if event not in ('open','os.listdir','os.scandir') or not args or isinstance(args[0],int):return
 try:p=Path(os.fsdecode(args[0])).resolve()
 except(TypeError,ValueError):return
 if p.is_relative_to(original):
  blocked.append(dict(event=event,path=str(p)));raise PermissionError('Original workspace access forbidden: '+str(p))
 if event=='open' and p.is_relative_to(checkout):
  mode=args[1]if len(args)>1 else None;flags=args[2]if len(args)>2 else 0
  writing=(isinstance(mode,str)and any(c in mode for c in 'wax+'))or(isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  (writes if writing else(reads if p.is_file()else missing)).add(str(p.relative_to(checkout)))
sys.addaudithook(hook)
try:(original/'README.md').read_bytes()
except PermissionError:tested=True
else:raise AssertionError('guard failed')
assert len(blocked)==1;blocked.clear();sys.dont_write_bytecode=True;sys.path[:0]=[str(checkout),str(checkout/'scripts')];os.chdir(checkout);sys.argv=[str(checkout/script),*extra];success=False
# Resource-only overrides installed after the access hook and before project imports.
import torch,threadpoolctl
thread_requests=[];old_torch_threads=torch.set_num_threads;old_pool=threadpoolctl.threadpool_limits
old_torch_threads(1);torch.set_num_interop_threads(1)
def cap_torch(n):
 thread_requests.append(dict(api='torch.set_num_threads',requested=n,actual=1));return old_torch_threads(1)
def cap_pool(limits=None,user_api=None):
 thread_requests.append(dict(api='threadpool_limits',requested=limits,actual=1));return old_pool(limits=1,user_api=user_api)
torch.set_num_threads=cap_torch;threadpoolctl.threadpool_limits=cap_pool
try:runpy.run_path(str(checkout/script),run_name='__main__');success=True
finally:record.write_text(json.dumps(dict(success=success,guard_self_test=tested,blocked_original_workspace_accesses=blocked,checkout_reads=sorted(reads),checkout_writes=sorted(writes),nonexistent_read_probes=sorted(missing),resource_thread_caps=thread_requests,final_torch_threads=torch.get_num_threads()),indent=2)+'\n')
