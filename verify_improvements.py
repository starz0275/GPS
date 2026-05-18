"""
代码改进验证清单 & 修复状态报告
此脚本用于验证所有改进是否已正确应用
"""
import sys
from pathlib import Path

def check_file_fix(filepath, check_func, description):
    """检查单个文件的修复状态"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        is_fixed = check_func(content)
        status = "[OK] 已修复" if is_fixed else "[X] 未修复"
        print(f"{status} | {filepath.name:30} | {description}")
        return is_fixed
    except FileNotFoundError:
        print(f"[X] 文件不存在 | {filepath.name:30} | {description}")
        return False
    except Exception as e:
        print(f"[X] 检查错误 | {filepath.name:30} | {str(e)[:40]}")
        return False


def main():
    print("\n" + "="*100)
    print("代码改进验证清单")
    print("="*100 + "\n")
    
    base_path = Path(__file__).parent
    checks = []
    
    # 1. train_tcn_model.py - Dropout移除
    checks.append(
        ("train_tcn_model.py", 
         lambda c: "layers.Dropout" not in c,
         "Dropout层已移除 (INT8兼容)")
    )
    
    # 2. train_tcn_model.py - MSE损失
    checks.append(
        ("train_tcn_model.py",
         lambda c: "loss='mse'" in c and "Huber" not in c,
         "损失函数已改为MSE")
    )
    
    # 3. build_tcn_model.py - 残差块投影层
    checks.append(
        ("build_tcn_model.py",
         lambda c: "res_project" in c or "project" in c,
         "残差块已添加投影层处理维度")
    )
    
    # 4. quantize_to_int8.py - 移除deprecated参数
    checks.append(
        ("quantize_to_int8.py",
         lambda c: "safe_mode=False" not in c,
         "load_model已移除deprecated参数")
    )
    
    # 5. simulate_tunnel.py - 索引边界检查
    checks.append(
        ("simulate_tunnel.py",
         lambda c: "real_row_idx >= len(df)" in c,
         "已添加索引越界检查")
    )
    
    # 6. 新文件 - 改进的量化管道
    checks.append(
        ("quantize_improved.py",
         lambda c: "verify_int8_compatibility" in c,
         "新增改进版量化管道")
    )
    
    # 7. 代码审查文档
    checks.append(
        ("CODE_REVIEW.py",
         lambda c: "改进优先级" in c,
         "已生成详细的代码审查报告")
    )
    
    # 执行检查
    results = {}
    for filename, check_func, description in checks:
        filepath = base_path / filename
        is_fixed = check_file_fix(filepath, check_func, description)
        results[filename] = is_fixed
    
    # 统计
    print("\n" + "="*100)
    total = len(results)
    fixed = sum(1 for v in results.values() if v)
    print(f"修复进度: {fixed}/{total} 项 ({100*fixed//total}%)")
    print("="*100 + "\n")
    
    if fixed == total:
        print("[OK] 所有改进已应用！")
        print("\n建议下一步行动:")
        print("  1. python train_tcn_model.py     # 重新训练（无Dropout，MSE损失）")
        print("  2. python quantize_improved.py   # 执行INT8量化")
        print("  3. python simulate_tunnel.py     # 验证隧道场景")
    else:
        print(f"[WARN] 仍有 {total - fixed} 项待修复")
        print("\n未修复的项目:")
        for filename, is_fixed in results.items():
            if not is_fixed:
                print(f"  - {filename}")
    
    return 0 if fixed == total else 1


if __name__ == "__main__":
    sys.exit(main())
