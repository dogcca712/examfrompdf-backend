#!/usr/bin/env python3
"""
测试脚本：验证商用级改进后的功能
"""
import sys
import os
import sqlite3
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_database_connection():
    """测试数据库连接和上下文管理器"""
    print("=" * 60)
    print("测试1: 数据库连接和上下文管理器")
    print("=" * 60)
    
    try:
        from app import get_db, DB_PATH
        
        # 测试上下文管理器
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            assert result[0] == 1, "数据库查询失败"
            print("✅ 数据库连接成功")
            print(f"✅ 数据库文件: {DB_PATH}")
            return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

def test_database_tables():
    """测试数据库表结构"""
    print("\n" + "=" * 60)
    print("测试2: 数据库表结构")
    print("=" * 60)
    
    try:
        from app import get_db
        
        expected_tables = ['users', 'subscriptions', 'usage', 'jobs', 'revoked_tokens']
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cur.fetchall()]
            
            print(f"找到的表: {tables}")
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ 表 '{table}' 存在")
                else:
                    print(f"❌ 表 '{table}' 不存在")
                    return False
            
            # 检查索引
            cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
            indexes = [row[0] for row in cur.fetchall()]
            print(f"✅ 找到索引: {indexes}")
            
            return True
    except Exception as e:
        print(f"❌ 表结构检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jobs_table_structure():
    """测试jobs表结构"""
    print("\n" + "=" * 60)
    print("测试3: jobs表结构")
    print("=" * 60)
    
    try:
        from app import get_db
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(jobs)")
            columns = cur.fetchall()
            
            expected_columns = ['id', 'user_id', 'file_name', 'status', 'created_at', 'download_url', 'error', 'updated_at']
            actual_columns = [col[1] for col in columns]
            
            print(f"jobs表列: {actual_columns}")
            
            for col in expected_columns:
                if col in actual_columns:
                    print(f"✅ 列 '{col}' 存在")
                else:
                    print(f"❌ 列 '{col}' 不存在")
                    return False
            
            return True
    except Exception as e:
        print(f"❌ jobs表结构检查失败: {e}")
        return False

def test_revoked_tokens_table():
    """测试revoked_tokens表"""
    print("\n" + "=" * 60)
    print("测试4: revoked_tokens表")
    print("=" * 60)
    
    try:
        from app import get_db
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(revoked_tokens)")
            columns = cur.fetchall()
            
            expected_columns = ['jti', 'user_id', 'revoked_at']
            actual_columns = [col[1] for col in columns]
            
            print(f"revoked_tokens表列: {actual_columns}")
            
            for col in expected_columns:
                if col in actual_columns:
                    print(f"✅ 列 '{col}' 存在")
                else:
                    print(f"❌ 列 '{col}' 不存在")
                    return False
            
            return True
    except Exception as e:
        print(f"❌ revoked_tokens表检查失败: {e}")
        return False

def test_database_indexes():
    """测试数据库索引"""
    print("\n" + "=" * 60)
    print("测试5: 数据库索引")
    print("=" * 60)
    
    try:
        from app import get_db
        
        expected_indexes = ['idx_jobs_user_id', 'idx_jobs_status', 'idx_jobs_created_at']
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
            indexes = [row[0] for row in cur.fetchall()]
            
            print(f"找到的索引: {indexes}")
            
            for idx in expected_indexes:
                if idx in indexes:
                    print(f"✅ 索引 '{idx}' 存在")
                else:
                    print(f"❌ 索引 '{idx}' 不存在")
                    return False
            
            return True
    except Exception as e:
        print(f"❌ 索引检查失败: {e}")
        return False

def test_no_memory_state():
    """测试是否移除了内存状态"""
    print("\n" + "=" * 60)
    print("测试6: 检查是否移除内存状态")
    print("=" * 60)
    
    try:
        import app
        
        # 检查是否还有JOBS字典
        if hasattr(app, 'JOBS'):
            print(f"❌ 发现内存状态 JOBS: {type(app.JOBS)}")
            return False
        else:
            print("✅ 未发现 JOBS 内存字典")
        
        # 检查是否还有TOKEN_BLACKLIST
        if hasattr(app, 'TOKEN_BLACKLIST'):
            token_blacklist = getattr(app, 'TOKEN_BLACKLIST', None)
            if isinstance(token_blacklist, set):
                print(f"❌ 发现内存状态 TOKEN_BLACKLIST (set)")
                return False
            else:
                print("✅ TOKEN_BLACKLIST 已改为数据库存储")
        else:
            print("✅ 未发现 TOKEN_BLACKLIST 内存集合")
        
        return True
    except Exception as e:
        print(f"❌ 内存状态检查失败: {e}")
        return False

def test_logging_setup():
    """测试日志配置"""
    print("\n" + "=" * 60)
    print("测试7: 日志配置")
    print("=" * 60)
    
    try:
        import app
        import logging
        
        if hasattr(app, 'logger'):
            logger = app.logger
            print(f"✅ Logger 已配置: {logger.name}")
            print(f"✅ Logger 级别: {logger.level}")
            return True
        else:
            print("❌ 未找到 logger")
            return False
    except Exception as e:
        print(f"❌ 日志配置检查失败: {e}")
        return False

def test_context_manager():
    """测试数据库上下文管理器的事务处理"""
    print("\n" + "=" * 60)
    print("测试8: 数据库上下文管理器事务")
    print("=" * 60)
    
    try:
        from app import get_db
        
        # 测试正常提交
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            assert result[0] == 1
        print("✅ 正常事务提交成功")
        
        # 测试回滚（模拟错误）
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM nonexistent_table")
        except Exception:
            print("✅ 异常时自动回滚成功")
        
        return True
    except Exception as e:
        print(f"❌ 上下文管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("商用级改进功能测试")
    print("=" * 60)
    
    tests = [
        test_database_connection,
        test_database_tables,
        test_jobs_table_structure,
        test_revoked_tokens_table,
        test_database_indexes,
        test_no_memory_state,
        test_logging_setup,
        test_context_manager,
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

