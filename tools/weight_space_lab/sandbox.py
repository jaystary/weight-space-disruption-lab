"""Unprivileged chroot plus syscall denials for generated NumPy programs."""
from __future__ import annotations
import ctypes
import errno
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import signal
import subprocess
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[2]
JAIL=ROOT/'editor-jail'

def build():
    import numpy
    JAIL.mkdir(exist_ok=True)
    # Only runtime libraries and explicitly allowlisted tensor snapshots enter jail.
    stdlib=Path(sysconfig.get_path('stdlib'))
    shutil.copytree(stdlib,JAIL/'usr/lib/python3.12',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','test','tests','dist-packages','site-packages'))
    numpy_dir=Path(numpy.__file__).parent
    shutil.copytree(numpy_dir,JAIL/'usr/lib/python3.12/site-packages/numpy',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','tests'))
    libs=numpy_dir.parent/'numpy.libs'
    if libs.exists(): shutil.copytree(libs,JAIL/'usr/lib/python3.12/site-packages/numpy.libs',dirs_exist_ok=True)
    sources=[Path(sys.executable).resolve(),Path('/usr/lib/x86_64-linux-gnu/libseccomp.so.2')]
    sources+=list(numpy_dir.rglob('*.so'))+list(libs.glob('*.so*'))
    found=set()
    for source in sources:
        p=subprocess.run(['ldd',str(source)],capture_output=True,text=True,check=False)
        for line in p.stdout.splitlines():
            for word in line.split():
                if word.startswith('/') and Path(word).is_file(): found.add(Path(word))
    for source in found:
        target=JAIL/str(source).lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    snapshots=JAIL/'snapshots';snapshots.mkdir(exist_ok=True)
    for source in (ROOT/'snapshots').iterdir():
        if source.suffix not in ('.npy','.json'): continue
        target=snapshots/source.name
        if not target.exists(): os.link(source,target)
        target.chmod(0o444)
    (JAIL/'work').mkdir(exist_ok=True)
    (JAIL/'work').chmod(0o777)
    (JAIL/'tmp').mkdir(exist_ok=True);(JAIL/'tmp').chmod(0o777)
    print(json.dumps({'runtime_ready':True,'tensor_files':len(list(snapshots.glob('*.npy')))}))

def child(script):
    # Load enforcement before chroot; never load credentials or corpus here.
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    lib.seccomp_release.argtypes=[ctypes.c_void_p]
    context=lib.seccomp_init(0x7fff0000)
    if not context: raise RuntimeError('seccomp_init failed')
    deny=('socket','socketpair','connect','bind','listen','accept','accept4','sendto','sendmsg','recvfrom','recvmsg',
          'execve','execveat','fork','vfork','clone','clone3','ptrace','process_vm_readv','process_vm_writev',
          'mount','umount2','pivot_root','chroot','setns','unshare','open_by_handle_at','bpf','perf_event_open',
          'kill','tkill','tgkill','reboot','kexec_load','keyctl','add_key','request_key')
    for name in deny:
        number=lib.seccomp_syscall_resolve_name(name.encode())
        if number>=0 and lib.seccomp_rule_add(context,0x00050000|errno.EPERM,number,0)!=0:
            raise RuntimeError('seccomp rule failed')
    os.environ.clear()
    os.environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','HOME':'/work','TMPDIR':'/tmp'})
    os.chroot(JAIL);os.chdir('/work')
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC,(1,1))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    os.closerange(3,1024)
    if lib.seccomp_load(context)!=0: raise RuntimeError('seccomp_load failed')
    lib.seccomp_release(context)
    sys.path[:]=['/work','/usr/lib/python3.12','/usr/lib/python3.12/lib-dynload','/usr/lib/python3.12/site-packages']
    runpy.run_path('/work/'+script,run_name='__main__')

def execute(code: str, directory: Path):
    directory.mkdir(parents=True,exist_ok=False)
    script=JAIL/'work/candidate.py'
    if script.exists(): script.chmod(0o644)
    script.write_text(code);script.chmod(0o444)
    patch=JAIL/'work/patch.npz'
    if patch.exists() or patch.is_symlink(): patch.unlink()
    with (directory/'stdout.log').open('xb') as output,(directory/'stderr.log').open('xb') as errors:
        process=subprocess.Popen([sys.executable,'-m','tools.weight_space_lab.sandbox','child','candidate.py'],
                                 cwd=ROOT,stdout=output,stderr=errors,start_new_session=True)
        try: rc=process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();rc=-9
    if patch.exists():
        if patch.is_symlink() or not patch.is_file() or patch.stat().st_size>16*1024**2:
            raise ValueError('unsafe patch output')
        shutil.copyfile(patch,directory/'patch.npz')
    (directory/'receipt.json').write_text(json.dumps({'returncode':rc,'patch':(directory/'patch.npz').exists()})+'\n')
    return rc

if __name__=='__main__':
    if sys.argv[1]=='build': build()
    elif sys.argv[1]=='child': child(sys.argv[2])
    else: raise SystemExit('unknown mode')
