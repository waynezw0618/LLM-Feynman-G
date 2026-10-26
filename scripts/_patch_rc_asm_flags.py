# -*- coding: utf-8 -*-
import ast, sys

p = r'scripts\rc_asm_run.py'
m = open(p, encoding='utf-8').read()

old_flag = ('    ap.add_argument("--wall_seeds", type=int, default=1, choices=[0, 1],\n'
            '                    help="P1d-B2 壁面特征程序化种子注入 (1=开, 默认; 0=关, 裸配置消融)")\n')
new_flag = ('    ap.add_argument("--wall_seeds", type=int, default=1, choices=[0, 1],\n'
            '                    help="P1d-B2 壁面特征程序化种子注入 (1=开, 默认; 0=关, 裸配置消融)")\n'
            '    # ===== v3a: G_INIT 鲁棒性 (A1 解析重试 + A3 few-shot) =====\n'
            '    ap.add_argument("--ginit_retry", type=int, default=2,\n'
            '                    help="G_INIT 解析率不达标时最大重试次数 (0=单次, 2=默认最多 3 次)")\n'
            '    ap.add_argument("--ginit_min_kept", type=int, default=3,\n'
            '                    help="G_INIT 累积解析成功数 >= 此值停止重试 (默认 3)")\n'
            '    ap.add_argument("--ginit_fewshot", type=int, default=1, choices=[0, 1],\n'
            '                    help="G_INIT prompt 注入 few-shot 合法 DSL 示例 (1=开, 默认; 0=关)")\n')
assert old_flag in m, 'wall_seeds flag not found'
m = m.replace(old_flag, new_flag, 1)

old_cfg = ('                        forbidden_ops=forbid_map.get(tgt, []),\n'
           '                        llm_ginit_only=bool(args.ginit_only),\n'
           '                        wall_seeds=bool(args.wall_seeds))\n')
new_cfg = ('                        forbidden_ops=forbid_map.get(tgt, []),\n'
           '                        llm_ginit_only=bool(args.ginit_only),\n'
           '                        wall_seeds=bool(args.wall_seeds),\n'
           '                        ginit_retry=int(args.ginit_retry),\n'
           '                        ginit_min_kept=int(args.ginit_min_kept),\n'
           '                        ginit_fewshot=bool(args.ginit_fewshot))\n')
assert old_cfg in m, 'LGFConfig block not found'
m = m.replace(old_cfg, new_cfg, 1)

old_sum = ('forbid_map=forbid_map, ginit_only=bool(args.ginit_only),\n'
           '                   wall_seeds=bool(args.wall_seeds),\n')
new_sum = ('forbid_map=forbid_map, ginit_only=bool(args.ginit_only),\n'
           '                   wall_seeds=bool(args.wall_seeds),\n'
           '                   ginit_retry=int(args.ginit_retry),\n'
           '                   ginit_min_kept=int(args.ginit_min_kept),\n'
           '                   ginit_fewshot=bool(args.ginit_fewshot),\n')
assert old_sum in m, 'summary block not found'
m = m.replace(old_sum, new_sum, 1)

open(p, 'w', encoding='utf-8', newline='').write(m)
ast.parse(m)
print('OK patched, len:', len(m))
