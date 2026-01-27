#!/usr/bin/env python3
"""
清空数据库所有数据，但保留表结构
用于重新开始，使用新的简化注册逻辑
"""

import sqlite3
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
# 支持通过环境变量指定数据库路径
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data.db"))

def reset_database():
    """清空数据库所有数据，但保留表结构"""
    if not DB_PATH.exists():
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return False
    
    # 确认操作
    print("=" * 80)
    print("⚠️  警告：此操作将清空数据库中的所有数据！")
    print("=" * 80)
    print(f"数据库路径: {DB_PATH}")
    print("\n将删除以下表中的所有数据：")
    print("  - users (用户)")
    print("  - jobs (任务)")
    print("  - usage (使用记录)")
    print("  - transactions (交易记录)")
    print("  - anon_usage (匿名使用记录)")
    print("  - anon_to_user (匿名用户关联)")
    print("  - guest_usage (访客使用记录)")
    print("  - registration_bonus (注册奖励)")
    print("  - revoked_tokens (撤销的token)")
    print("\n⚠️  表结构将保留，但所有数据将被删除！")
    print("=" * 80)
    
    confirm = input("\n确认要清空数据库吗？请输入 'YES' 继续: ")
    if confirm != "YES":
        print("❌ 操作已取消")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # 获取所有表名
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        
        print(f"\n开始清空 {len(tables)} 个表...")
        
        # 禁用外键检查（SQLite默认启用）
        cur.execute("PRAGMA foreign_keys = OFF")
        
        # 清空每个表
        deleted_counts = {}
        for table in tables:
            try:
                cur.execute(f"DELETE FROM {table}")
                count = cur.rowcount
                deleted_counts[table] = count
                print(f"  ✅ {table}: 删除了 {count} 条记录")
            except Exception as e:
                print(f"  ❌ {table}: 删除失败 - {e}")
        
        # 重置自增ID（SQLite使用AUTOINCREMENT时需要）
        for table in tables:
            try:
                # 检查表是否有AUTOINCREMENT列
                cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                sql = cur.fetchone()
                if sql and 'AUTOINCREMENT' in sql[0]:
                    cur.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
                    print(f"  ✅ 重置了 {table} 的自增ID")
            except Exception as e:
                # 如果没有sqlite_sequence表或不需要重置，忽略错误
                pass
        
        # 重新启用外键检查
        cur.execute("PRAGMA foreign_keys = ON")
        
        # 提交事务
        conn.commit()
        
        print("\n" + "=" * 80)
        print("✅ 数据库清空完成！")
        print("=" * 80)
        print("\n删除统计：")
        total_deleted = 0
        for table, count in deleted_counts.items():
            print(f"  - {table}: {count} 条记录")
            total_deleted += count
        print(f"\n总计删除: {total_deleted} 条记录")
        print("\n表结构已保留，可以重新开始使用。")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ 清空数据库时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = reset_database()
    exit(0 if success else 1)

