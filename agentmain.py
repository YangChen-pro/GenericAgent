"""Legacy GenericAgent entrypoint."""

import glob
import json
import os
import sys
import threading
import time

from ga import consume_file
from ga_config import load_tool_schema as _load_tool_schema, script_dir
from ga_runtime import GeneraticAgent, GenericAgent

TOOLS_SCHEMA = _load_tool_schema()

def load_tool_schema(suffix=""):
    """Compatibility wrapper used by existing frontends."""
    global TOOLS_SCHEMA
    TOOLS_SCHEMA = _load_tool_schema(suffix)
    return TOOLS_SCHEMA


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', metavar='IODIR', help='一次性任务模式，先看subagent.md')
    parser.add_argument('--func', metavar='PROMPT_FILE', help='纯函数模式：读prompt文件→结果写prompt.out.txt→退出')
    parser.add_argument('--reflect', metavar='SCRIPT', help='反射模式：加载监控脚本，check()触发时发任务')
    parser.add_argument('--input', help='prompt')
    parser.add_argument('--history', help='history json file')
    parser.add_argument('--llm_no', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nobg', action='store_true')
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--no-user-tools', action='store_true')
    parser.add_argument('--no-memory-write', action='store_true')
    args, _unknown = parser.parse_known_args()
    _extra_args = dict(zip([k.lstrip('-') for k in _unknown[::2]], _unknown[1::2])) if _unknown else {}

    if (args.func or args.task) and not args.nobg:
        import subprocess, platform
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:]] + ['--nobg']
        if args.task:
            d = os.path.join(script_dir, f'temp/{args.task}'); os.makedirs(d, exist_ok=True)
            out = open(os.path.join(d, 'stdout.log'), 'w', encoding='utf-8')
            err = open(os.path.join(d, 'stderr.log'), 'w', encoding='utf-8')
        else: out, err = subprocess.DEVNULL, subprocess.DEVNULL
        p = subprocess.Popen(cmd, cwd=script_dir,
            creationflags=0x08000000 if platform.system() == 'Windows' else 0,
            stdout=out, stderr=err)
        print('PID:', p.pid); sys.exit(0)

    agent = GenericAgent()
    if args.nolog: agent.log_path = False
    agent.next_llm(args.llm_no)
    agent.verbose = args.verbose
    threading.Thread(target=agent.run, daemon=True).start()

    histfile = args.history
    if args.task:
        agent.task_dir = d = os.path.join(script_dir, f'temp/{args.task}'); nround = ''
        infile = os.path.join(d, 'input.txt'); outfile = f'{d}/output{nround}.txt'
        if args.input:
            os.makedirs(d, exist_ok=True)
            [os.remove(f) for f in glob.glob(os.path.join(d, 'output*.txt'))]
            with open(infile, 'w', encoding='utf-8') as f: f.write(args.input)
        histfile = histfile or os.path.join(d, '_history.json')
    elif args.func:
        infile = args.func; outfile = os.path.splitext(args.func)[0] + '.out.txt'

    if histfile and os.path.isfile(histfile): agent.llmclient.backend.history = json.loads(open(histfile, encoding='utf-8').read())

    if args.func or args.task:
        agent.peer_hint = False
        with open(infile, encoding='utf-8') as f: raw = f.read()
        while True:
            dq = agent.put_task(raw, source='func' if args.func else 'task')
            while 'done' not in (item := dq.get(timeout=2200)):
                if 'next' in item:
                    with open(outfile, 'w', encoding='utf-8') as f: f.write(item.get('next', ''))
            with open(outfile, 'w', encoding='utf-8') as f: f.write(item['done'] + '\n\n[ROUND END]\n')
            if not args.task: break
            consume_file(d, '_stop')  # 已经成功停下来了，避免打断下次reply
            for _ in range(300):  # 等reply.txt，10分钟超时
                time.sleep(2)
                if (raw := consume_file(d, 'reply.txt')): break
            else: break
            nround = nround + 1 if isinstance(nround, int) else 1
            outfile = f'{d}/output{nround}.txt'
    elif args.reflect:
        agent.peer_hint = False
        import importlib.util
        spec = importlib.util.spec_from_file_location('reflect_script', args.reflect)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        if hasattr(mod, 'init'): mod.init(_extra_args)
        _mt = os.path.getmtime(args.reflect)
        print(f'[Reflect] loaded {args.reflect}' + (f' args={_extra_args}' if _extra_args else ''))
        try:
            import frontends.hub as hub
            def _hub_put(t):
                if agent.is_running: return {'error': 'busy', 'code': 'busy'}
                agent.put_task(t, source='hub')
            hub.connect(agent, f'reflect/{os.path.splitext(os.path.basename(args.reflect))[0]}', put_task=_hub_put)
        except Exception: pass
        while True:
            if os.path.getmtime(args.reflect) != _mt:
                try:
                    spec.loader.exec_module(mod); _mt = os.path.getmtime(args.reflect)
                    if hasattr(mod, 'init'): mod.init(_extra_args)
                    print('[Reflect] reloaded')
                except Exception as e: print(f'[Reflect] reload error: {e}')
            try: task = mod.check()
            except Exception as e: 
                print(f'[Reflect] check() error: {e}'); task = None
            if task and task == '/exit': break
            if task:
                print(f'[Reflect] triggered: {task[:80]}')
                dq = agent.put_task(task, source='reflect')
                try:
                    while 'done' not in (item := dq.get(timeout=2200)): pass
                    result = item['done']
                    print(result)
                except Exception as e:
                    if getattr(mod, 'ONCE', False): raise
                    print(f'[Reflect] drain error: {e}'); result = f'[ERROR] {e}'
                log_dir = os.path.join(script_dir, 'temp/reflect_logs'); os.makedirs(log_dir, exist_ok=True)
                script_name = os.path.splitext(os.path.basename(args.reflect))[0]
                open(os.path.join(log_dir, f'{script_name}_{datetime.now():%Y-%m-%d}.log'), 'a', encoding='utf-8').write(f'[{datetime.now():%m-%d %H:%M}]\n{result}\n\n')
                if (on_done := getattr(mod, 'on_done', None)):
                    try: on_done(result)
                    except Exception as e: print(f'[Reflect] on_done error: {e}')
                if getattr(mod, 'ONCE', False): print('[Reflect] ONCE=True, exiting.'); break
            time.sleep(getattr(mod, 'INTERVAL', 5))
    else:
        try: import readline
        except Exception: pass
        agent.inc_out = True
        if sys.stdout.isatty():
            try: model = agent.get_llm_name(model=True) or '?'
            except Exception: model = '?'
            try:
                sys.stdout.write(f'\x1b[92m✦\x1b[0m \x1b[1mGenericAgent\x1b[0m '
                                 f'\x1b[90m· cli · model:\x1b[0m {model}\n')
                sys.stdout.flush()
            except Exception: pass
        while True:
            q = input('> ').strip()
            if not q: continue
            try:
                dq = agent.put_task(q, source='user')
                while True:
                    item = dq.get()
                    if 'next' in item: print(item['next'], end='', flush=True)
                    if 'done' in item: print(); break
            except KeyboardInterrupt:
                agent.abort(); print('\n[Interrupted]')
