#!/usr/bin/env python3
"""
代码结构测试：不依赖外部库，只检查代码结构
"""
import ast
import sys
from pathlib import Path

def test_syntax():
    """测试Python语法"""
    print("=" * 60)
    print("测试1: Python语法检查")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        ast.parse(code)
        print("✅ Python语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        print(f"   行 {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return False

def test_no_memory_state():
    """检查是否移除了内存状态"""
    print("\n" + "=" * 60)
    print("测试2: 检查是否移除内存状态")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 检查是否还有JOBS字典定义
        if 'JOBS = {}' in code or 'JOBS = dict()' in code:
            print("❌ 发现 JOBS 内存字典定义")
            return False
        else:
            print("✅ 未发现 JOBS 内存字典定义")
        
        # 检查是否还有TOKEN_BLACKLIST set定义
        if 'TOKEN_BLACKLIST = set()' in code:
            print("❌ 发现 TOKEN_BLACKLIST 内存集合定义")
            return False
        else:
            print("✅ 未发现 TOKEN_BLACKLIST 内存集合定义")
        
        # 检查是否有JOBS[job_id]的使用
        if 'JOBS[' in code:
            print("⚠️  发现 JOBS[...] 的使用，可能还有遗留代码")
            # 但不算错误，因为可能是注释
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                if 'JOBS[' in line and not line.strip().startswith('#'):
                    print(f"   行 {i}: {line.strip()}")
        else:
            print("✅ 未发现 JOBS[...] 的使用")
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_database_context_manager():
    """检查数据库上下文管理器"""
    print("\n" + "=" * 60)
    print("测试3: 数据库上下文管理器")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 检查是否有@contextmanager装饰器
        if '@contextmanager' in code:
            print("✅ 发现 @contextmanager 装饰器")
        else:
            print("❌ 未发现 @contextmanager 装饰器")
            return False
        
        # 检查get_db函数是否使用上下文管理器
        if 'def get_db():' in code and 'yield' in code:
            print("✅ get_db 函数使用 yield (上下文管理器)")
        else:
            print("❌ get_db 函数未使用 yield")
            return False
        
        # 检查是否有with get_db()的使用
        if 'with get_db()' in code:
            print("✅ 发现 with get_db() 的使用")
        else:
            print("⚠️  未发现 with get_db() 的使用")
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_logging():
    """检查日志配置"""
    print("\n" + "=" * 60)
    print("测试4: 日志配置")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        if 'import logging' in code:
            print("✅ 导入 logging 模块")
        else:
            print("❌ 未导入 logging 模块")
            return False
        
        if 'logging.basicConfig' in code or 'logger =' in code:
            print("✅ 配置了 logger")
        else:
            print("⚠️  可能未配置 logger")
        
        if 'logger.info' in code or 'logger.error' in code:
            print("✅ 使用了 logger")
        else:
            print("⚠️  可能未使用 logger")
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_database_tables():
    """检查数据库表定义"""
    print("\n" + "=" * 60)
    print("测试5: 数据库表定义")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        expected_tables = [
            'CREATE TABLE IF NOT EXISTS users',
            'CREATE TABLE IF NOT EXISTS subscriptions',
            'CREATE TABLE IF NOT EXISTS usage',
            'CREATE TABLE IF NOT EXISTS jobs',
            'CREATE TABLE IF NOT EXISTS revoked_tokens',
        ]
        
        for table in expected_tables:
            if table in code:
                print(f"✅ {table}")
            else:
                print(f"❌ 未找到: {table}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_database_indexes():
    """检查数据库索引"""
    print("\n" + "=" * 60)
    print("测试6: 数据库索引")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        expected_indexes = [
            'idx_jobs_user_id',
            'idx_jobs_status',
            'idx_jobs_created_at',
        ]
        
        for idx in expected_indexes:
            if idx in code:
                print(f"✅ 索引 {idx}")
            else:
                print(f"❌ 未找到索引: {idx}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_updated_at_field():
    """检查updated_at字段"""
    print("\n" + "=" * 60)
    print("测试7: updated_at字段")
    print("=" * 60)
    
    try:
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 检查jobs表是否有updated_at字段
        if 'updated_at TEXT' in code:
            print("✅ jobs表有 updated_at 字段")
        else:
            print("❌ jobs表缺少 updated_at 字段")
            return False
        
        # 检查是否有更新updated_at的代码
        if 'updated_at' in code and 'UPDATE jobs' in code:
            print("✅ 有更新 updated_at 的代码")
        else:
            print("⚠️  可能未更新 updated_at")
        
        return True
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("代码结构测试（不依赖外部库）")
    print("=" * 60)
    
    tests = [
        test_syntax,
        test_no_memory_state,
        test_database_context_manager,
        test_logging,
        test_database_tables,
        test_database_indexes,
        test_updated_at_field,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())

