from pathlib import Path
import os,sys,json,runpy
checkout=Path(sys.argv[1]).resolve();original=Path(sys.argv[2]).resolve();prior=Path(sys.argv[3]).resolve();record=Path(sys.argv[4]);script=sys.argv[5];reads=set();writes=set();blocked=[];missing=set();output=checkout/'results/native_input_observability/report'
def hook(event,args):
 if event not in ('open','os.listdir','os.scandir')or not args or isinstance(args[0],int):return
 try:p=Path(os.fsdecode(args[0])).resolve()
 except(TypeError,ValueError):return
 if p.is_relative_to(original)or p.is_relative_to(prior):blocked.append(dict(event=event,path=str(p)));raise PermissionError('Non-checkout project access forbidden: '+str(p))
 if event=='open'and p.is_relative_to(checkout):
  mode=args[1]if len(args)>1 else None;flags=args[2]if len(args)>2 else 0;writing=(isinstance(mode,str)and any(x in mode for x in 'wax+'))or(isinstance(flags,int)and bool(flags&(os.O_WRONLY|os.O_RDWR)))
  if writing and not p.is_relative_to(output):raise PermissionError('Shared input is read-only: '+str(p))
  (writes if writing else(reads if p.is_file()else missing)).add(str(p.relative_to(checkout)))
sys.addaudithook(hook)
for forbidden in(original,prior):
 try:(forbidden/'README.md').read_bytes()
 except PermissionError:pass
 else:raise AssertionError('Read guard self-test failed')
assert len(blocked)==2;blocked.clear();sys.dont_write_bytecode=True;sys.path[:0]=[str(checkout),str(checkout/'scripts')];os.chdir(checkout);sys.argv=[str(checkout/script)];success=False
try:runpy.run_path(str(checkout/script),run_name='__main__');success=True
finally:record.write_text(json.dumps(dict(success=success,both_forbidden_roots_self_tested=True,blocked_forbidden_accesses=blocked,checkout_reads=sorted(reads),checkout_writes=sorted(writes),nonexistent_read_probes=sorted(missing)),indent=2)+'\n')
