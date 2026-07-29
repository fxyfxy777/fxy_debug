#!/usr/bin/env python
import os, re, sys, time, base64, random, string, shutil, argparse, tempfile
from subprocess import run, Popen, PIPE

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('-e', '--exclusive', action='store_true', help='Execute only when all GPUs are free')
parser.add_argument('--ip', action='store_true', help='Print ip and exit')
backend_map = {"1": "mpirun", "2": "ssh", "3": "sshrun"}
parser.add_argument('-b', '--backend', choices=sum(backend_map.items(), ()), help='Select MPI backend')
parser.add_argument('-r', '--rank', help='Specify rank range')
parser.add_argument('-f', '--file', help='Specify bash script to execute')
parser.add_argument('-c', '--code', help='Specify bash code to execute')
parser.add_argument('-m', '--master', help='Use this master instead of rank 0')
parser.add_argument('-n', '--nnodes', help='Use this nnodes instead of the number of ranks', type=int)
parser.add_argument('-s', '--start-rank', help='Count ranks starting from this number', type=int, default=0)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()

if (args.file is None) and (args.code is None) and (not args.command) and (not args.ip):
    parser.print_help()
    exit()
if (args.file is not None) and (args.code is not None):
    parser.error("'--file' and '--code' cannot be used together")
if args.backend is not None:
    args.backend = backend_map.get(args.backend, args.backend)

re_rank_range = re.compile(r'^(\d+)-(\d+)$')
re_bash_safe = re.compile(r'^[\w\-\./]*$')

tempdir = '/dev/shm' if os.path.exists('/dev/shm') else tempfile.gettempdir()
with open('/root/paddlejob/workspace/hostfile') as f:
    ip_list = [line.split(maxsplit=1)[0] for line in f if line.strip()]
cur_rank = int(os.environ['PADDLE_TRAINER_ID'])
local_ip = ip_list[cur_rank]

try:  # set line width for mpirun.py
    os.environ.setdefault('LW', str(max(os.get_terminal_size().columns - 22, 127)))
except OSError:
    pass

def parse_rank(s, split=True):
    if split and (i := s.find('/')) != -1:
        assert s.find('/', i + 1) == -1, f"at most one '/' is allowed: {s}"
        accept = parse_rank(s[:i], split=False) or set(range(len(ip_list)))
        reject = parse_rank(s[i + 1:], split=False)
        return sorted(accept - reject)
    ranks = set()
    for item in s.split(','):
        if item.isdigit():
            ranks.add(int(item))
        elif m := re_rank_range.match(item):
            s, e = int(m.group(1)), int(m.group(2))
            assert e >= s, 'invalid rank range: %r' % item
            ranks.update(range(s, e + 1))
        elif s:
            raise ValueError('invalid rank expression: %r' % item)
    return sorted(ranks) if split else ranks

def gen_uid():
    n = int.from_bytes(os.urandom(16), 'little')
    chars = string.ascii_letters + string.digits
    indices = [(n := n // 62) % 62 for _ in range(8)]
    return ''.join(chars[i] for i in indices)

def wrap_arg(s):
    if re_bash_safe.match(s):
        return s
    if "'" not in s:
        return f"'{s}'"
    items = s.split("'")
    return r"\'".join(wrap_arg(item) for item in items)

if args.rank is None:
    ranks = []
    script = ['export LSHRUN_RANK=${PADDLE_TRAINER_ID} LSHRUN_NNODES=0']
else:
    ranks = parse_rank(args.rank)
    rank_map = ["''"] * (max(ranks) + 1)
    for i, rank in enumerate(ranks, args.start_rank):
        rank_map[rank] = str(i)
    master = ip_list[ranks[0]] if args.master is None else args.master
    nnodes = len(ranks) if args.nnodes is None else args.nnodes
    script = [f'export LSHRUN_RANK_MAP=({" ".join(rank_map)})',
              'export LSHRUN_RANK=${LSHRUN_RANK_MAP[PADDLE_TRAINER_ID]}', 'if [ -z $LSHRUN_RANK ]; then exit; fi',
              f'export LSHRUN_MASTER={master} LSHRUN_NNODES={nnodes}']

if args.ip:
    for rank in ranks:
        print(f'[ {rank} ] {ip_list[rank]}')
    exit()

sshrun_path = os.path.join(os.path.dirname(__file__), 'mpirun.py')
backend = args.backend or "ssh" * (len(ranks) == 1) or "sshrun" * os.path.exists(sshrun_path) or "mpirun"
assert backend != "ssh" or len(ranks) == 1, f"ssh only supports one rank, got {ranks}"

def run_script(script, method=None, **kwargs):
    if backend != "mpirun":
        script = [f'cd {wrap_arg(os.getcwd())} 2>/dev/null'] + script        
    tmp = f'/tmp/grep_{gen_uid()}'
    script = '\n'.join([f'rm -f {tmp}'] + script)
    print(f"{'-' * 80}\n{script}\n{'-' * 80}") if args.verbose else ()
    script = base64.b64encode(script.encode()).decode()
    script = f"__import__('os').system('echo\\x20{script}|base64\\x20-d>{tmp}&&sh\\x20{tmp}')"
    if backend == "ssh":
        cmd = ['ssh', f'root@{ip_list[ranks[0]]}', f'python -uc "{script}"']
    else:
        host_arg = ('--host', ','.join(ip_list[t] for t in ranks)) if ranks else ()
        if backend == "sshrun":
            cmd = ['python', sshrun_path, *host_arg, f'python -uc "{script}"']
        else:
            cmd = ['mpirun', *host_arg, f'python -uc {script}']
    return os.execvp(cmd[0], cmd) if method is None else method(cmd, **kwargs)

query_smi = "_smi=`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv | tail -n +2 | tr -d ' ' | tr ',' ' '`"
print_smi_fmt = f"[ %d ]{'%5s' * 8} |{'%10s' * 8}"
print_smi = f"printf '{print_smi_fmt}\n'"
print_smi_color = f"printf '\033[1;31m{print_smi_fmt}\033[0m\n'"
smi_arg = ''' $PADDLE_TRAINER_ID `awk '{print $1}' <<< "$_smi"` `awk '{print $2}' <<< "$_smi"`'''
check_smi = '''_check_smi() {
  if [ `grep ^0% <<< "$_smi" | wc -l` -ne 8 ]; then return 1; fi
  while IFS= read -r line; do mem=`awk '{print $2}' <<< "$line"`; mem=${mem%MiB};
  if [ $mem -gt 4 ]; then return 1; fi; done <<< "$_smi"; }'''
kmp_get_pids = '''while read line; do
  IFS=' ' read -r pid cmd <<< "$line"
  if [[ $line =~ liangshuhao ]]; then echo "$line"; _pids="$_pids $pid"
  elif realpath /proc/${pid}/cwd 2>/dev/null | grep liangshuhao >/dev/null
  then echo "$line"; _pids="$_pids $pid"; fi
done <<< $(pgrep python | grep -vE "^${PPID}$" | xargs -r ps -o pid,cmd | grep -vE "lshrun|mpirun")
if [ -z "$_pids" ]; then echo "No my python found"; else'''
kmp_query_kill = '''echo -n 'Comfirm kill [y/n]? '; read choice
  if [[ $choice =~ ^[Yy]$ ]]; then kill -9 $_pids; echo "All killed"
  else echo "Abort"; fi; fi'''

is_builtin = (args.file is None) and (args.code is None) and args.command[0] in ('smi', 'get', 'put', 'kmp')
if args.exclusive and not is_builtin:
    script += [query_smi, check_smi, 'if ! _check_smi; then', print_smi_color + smi_arg, 'exit; fi']

if (args.file is not None) or (args.code is not None):
    if args.command:
        script.append('set -- ' + ' '.join(wrap_arg(item) for item in args.command))
    if args.file is not None:
        with open(args.file) as f:
            script.append(f.read())
    else:
        script.append(args.code)
    run_script(script)

if args.command[0] == 'smi':
    script += [query_smi, print_smi + smi_arg]
    num_rows = cur_row = len(ranks) if ranks else len(ip_list)
    row_map = {rank: i for i, rank in enumerate(ranks)} if ranks else list(range(num_rows))
    lines = [None] * num_rows

    if is_tty := sys.stdout.isatty():
        os.environ["PYTHONUNBUFFERED"] = "1"
    p = run_script(script, method=Popen, stdout=PIPE, encoding='utf-8')
    print(end="\n" * cur_row) if is_tty else ()

    try:
        for line in p.stdout:
            if m := re.search(r'\[ (\d+) ]', line):
                row, line = row_map[int(m.group(1))], line.rstrip()
                lines[row] = line
                if is_tty:
                    delta = row - cur_row
                    move = f"\033[{-delta}A" if delta < 0 else f"\033[{delta}B"
                    print(end=f"{move}\r{line}", flush=True)
                    cur_row = row
    except KeyboardInterrupt:
        pass

    if is_tty:
        print(end=f"\033[{num_rows - cur_row}B\r")
    else:
        print(*lines, sep="\n")

elif args.command[0] == 'get':
    if len(ranks) != 1:
        parser.error("'get' only supports one rank at a time")
    sub_parser = argparse.ArgumentParser(usage='%(prog)s get [-h] [-o OUTPUT] path')
    sub_parser.add_argument('path')
    sub_parser.add_argument('-o', '--output', help='Specify output path')
    sub_args = sub_parser.parse_args(args.command[1:])

    # run remote send
    path = os.path.realpath(sub_args.path)
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    temp_dir = os.path.join(tempdir, 'lshrun_%08x' % random.getrandbits(32))
    os.mkdir(temp_dir)
    cmd = f'tar c -C {wrap_arg(dirname)} {wrap_arg(basename)} | ssh root@{local_ip} "tar x -C {temp_dir}"'
    run_script(script + [cmd], method=run)

    # rename received file
    output = sub_args.path if sub_args.output is None else sub_args.output
    shutil.move(os.path.join(temp_dir, basename), output)
    os.rmdir(temp_dir)

elif args.command[0] == 'put':
    sub_parser = argparse.ArgumentParser(usage='%(prog)s put [-h] [-x] path [path ...]')
    sub_parser.add_argument('path', nargs='+')
    sub_parser.add_argument('-x', '--replace', action='store_true', help='Delete the remote file before sending')
    sub_args = sub_parser.parse_args(args.command[1:])

    if cur_rank in ranks:
        ranks.remove(cur_rank)
        exit() if not ranks else ()

    # tar local files into a temp file
    temp_path = os.path.join(tempdir, 'lshrun_%08x.tar' % random.getrandbits(32))
    abs_files = [os.path.realpath(file) for file in sub_args.path]
    time_0 = time.time()
    run(['tar', '-Pcf', temp_path] + abs_files, check=True)
    size = os.stat(temp_path).st_size
    if (duration := (time_1 := time.time()) - time_0) > 1:
        print(f'pack {round(size / 1e9, 3)} GB in {round(duration, 3)} sec')

    # run remote fetch
    repl = ['rm -rf ' + ' '.join(wrap_arg(file) for file in abs_files)] if sub_args.replace else []
    run_script(script + repl + [f'ssh root@{local_ip} "cat {temp_path}" | tar x -P'], method=run)
    if (duration := time.time() - time_1) > 2:
        print(f'send {round(size / 1e9, 3)} GB in {round(duration, 3)} sec')

    os.remove(temp_path)

elif args.command[0] == 'kmp':
    sub_parser = argparse.ArgumentParser(usage='%(prog)s kmp [-h] [-q]')
    sub_parser.add_argument('-q', '--query', action='store_true', help='Query before killing')
    sub_args = sub_parser.parse_args(args.command[1:])
    if sub_args.query and len(ranks) != 1:
        parser.error("'kmp' with '--query' only supports one rank at a time")
    script.append(kmp_get_pids)
    script.append(kmp_query_kill if sub_args.query else 'kill -9 $_pids; fi')
    run_script(script)

else:
    script.append(' '.join(wrap_arg(item) for item in args.command))
    run_script(script)