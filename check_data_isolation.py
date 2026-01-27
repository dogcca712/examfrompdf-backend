#!/usr/bin/env python3
"""
扫描数据库，检查是否存在数据隔离问题：
1. 检查是否有重复的job_id（理论上不应该存在）
2. 检查是否有同一个job被多个user_id关联
3. 检查匿名jobs（user_id=NULL）的device_fingerprint分布
4. 检查是否有user_id不匹配但可能被错误访问的情况
"""

import sqlite3
import os
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
# 支持通过环境变量指定数据库路径，默认使用当前目录的data.db
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data.db"))

def check_data_isolation():
    """检查数据隔离问题"""
    if not DB_PATH.exists():
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    print("=" * 80)
    print("数据库数据隔离检查")
    print("=" * 80)
    
    # 1. 检查是否有重复的job_id（理论上不应该存在）
    print("\n1. 检查重复的job_id...")
    cur.execute("""
        SELECT id, COUNT(*) as count 
        FROM jobs 
        GROUP BY id 
        HAVING COUNT(*) > 1
    """)
    duplicate_jobs = cur.fetchall()
    if duplicate_jobs:
        print(f"❌ 发现 {len(duplicate_jobs)} 个重复的job_id:")
        for row in duplicate_jobs:
            print(f"   - job_id: {row['id']}, 出现次数: {row['count']}")
    else:
        print("✅ 没有发现重复的job_id")
    
    # 2. 检查匿名jobs（user_id=NULL）的device_fingerprint分布
    print("\n2. 检查匿名jobs的device_fingerprint分布...")
    cur.execute("""
        SELECT device_fingerprint, COUNT(*) as job_count, 
               GROUP_CONCAT(DISTINCT id) as job_ids
        FROM jobs 
        WHERE user_id IS NULL 
        GROUP BY device_fingerprint
        HAVING COUNT(*) > 1
        ORDER BY job_count DESC
        LIMIT 20
    """)
    anon_fingerprints = cur.fetchall()
    if anon_fingerprints:
        print(f"⚠️  发现 {len(anon_fingerprints)} 个device_fingerprint有多个匿名jobs:")
        for row in anon_fingerprints:
            job_ids = row['job_ids'].split(',')
            print(f"   - device_fingerprint: {row['device_fingerprint'][:16]}...")
            print(f"     jobs数量: {row['job_count']}")
            print(f"     job_ids: {', '.join(job_ids[:5])}{'...' if len(job_ids) > 5 else ''}")
    else:
        print("✅ 匿名jobs的device_fingerprint分布正常")
    
    # 3. 检查是否有同一个device_fingerprint关联到不同的user_id
    print("\n3. 检查device_fingerprint与user_id的关联...")
    cur.execute("""
        SELECT device_fingerprint, 
               COUNT(DISTINCT user_id) as user_count,
               GROUP_CONCAT(DISTINCT user_id) as user_ids,
               COUNT(*) as job_count
        FROM jobs 
        WHERE device_fingerprint IS NOT NULL
        GROUP BY device_fingerprint
        HAVING COUNT(DISTINCT user_id) > 1
        ORDER BY user_count DESC
        LIMIT 20
    """)
    fingerprint_users = cur.fetchall()
    if fingerprint_users:
        print(f"⚠️  发现 {len(fingerprint_users)} 个device_fingerprint关联到多个user_id:")
        for row in fingerprint_users:
            user_ids = [uid for uid in row['user_ids'].split(',') if uid]
            print(f"   - device_fingerprint: {row['device_fingerprint'][:16]}...")
            print(f"     关联的user_ids: {user_ids}")
            print(f"     jobs数量: {row['job_count']}")
            
            # 检查这些jobs的详细信息
            cur.execute("""
                SELECT id, user_id, file_name, created_at, status
                FROM jobs 
                WHERE device_fingerprint = ?
                ORDER BY created_at DESC
            """, (row['device_fingerprint'],))
            jobs = cur.fetchall()
            for job in jobs[:3]:  # 只显示前3个
                print(f"       - job_id: {job['id'][:8]}..., user_id: {job['user_id']}, file: {job['file_name'][:30]}")
    else:
        print("✅ device_fingerprint与user_id的关联正常")
    
    # 4. 检查是否有user_id不为NULL但可能被错误访问的情况
    print("\n4. 检查认证用户jobs的user_id分布...")
    cur.execute("""
        SELECT user_id, COUNT(*) as job_count
        FROM jobs 
        WHERE user_id IS NOT NULL
        GROUP BY user_id
        ORDER BY job_count DESC
        LIMIT 10
    """)
    user_jobs = cur.fetchall()
    print(f"   前10个用户的jobs数量:")
    for row in user_jobs:
        print(f"   - user_id: {row['user_id']}, jobs数量: {row['job_count']}")
    
    # 5. 检查是否有可疑的数据：同一个job_id但user_id不同（理论上不应该存在）
    print("\n5. 检查可疑数据：同一个job可能被多个用户看到...")
    cur.execute("""
        SELECT j1.id, j1.user_id as user_id_1, j2.user_id as user_id_2,
               j1.device_fingerprint, j1.file_name, j1.created_at
        FROM jobs j1
        JOIN jobs j2 ON j1.id = j2.id
        WHERE j1.user_id != j2.user_id 
           OR (j1.user_id IS NULL AND j2.user_id IS NOT NULL)
           OR (j1.user_id IS NOT NULL AND j2.user_id IS NULL)
    """)
    suspicious_jobs = cur.fetchall()
    if suspicious_jobs:
        print(f"❌ 发现 {len(suspicious_jobs)} 个可疑的jobs（同一个job_id但user_id不同）:")
        for row in suspicious_jobs:
            print(f"   - job_id: {row['id']}")
            print(f"     user_id_1: {row['user_id_1']}, user_id_2: {row['user_id_2']}")
            print(f"     file_name: {row['file_name']}")
    else:
        print("✅ 没有发现可疑的jobs（同一个job_id但user_id不同）")
    
    # 6. 检查匿名jobs是否可能被错误关联
    print("\n6. 检查匿名jobs的详细情况...")
    cur.execute("""
        SELECT COUNT(*) as total_anon_jobs,
               COUNT(DISTINCT device_fingerprint) as unique_fingerprints
        FROM jobs 
        WHERE user_id IS NULL
    """)
    anon_stats = cur.fetchone()
    if anon_stats:
        print(f"   - 匿名jobs总数: {anon_stats['total_anon_jobs']}")
        print(f"   - 唯一的device_fingerprint数量: {anon_stats['unique_fingerprints']}")
        if anon_stats['total_anon_jobs'] > 0:
            avg_jobs_per_fingerprint = anon_stats['total_anon_jobs'] / anon_stats['unique_fingerprints']
            print(f"   - 平均每个fingerprint的jobs数: {avg_jobs_per_fingerprint:.2f}")
    
    # 7. 检查是否有jobs的user_id和device_fingerprint都不匹配的情况
    print("\n7. 检查数据完整性...")
    cur.execute("""
        SELECT COUNT(*) as total_jobs,
               COUNT(CASE WHEN user_id IS NULL THEN 1 END) as anon_jobs,
               COUNT(CASE WHEN user_id IS NOT NULL THEN 1 END) as user_jobs,
               COUNT(CASE WHEN device_fingerprint IS NULL THEN 1 END) as no_fingerprint
        FROM jobs
    """)
    stats = cur.fetchone()
    if stats:
        print(f"   - 总jobs数: {stats['total_jobs']}")
        print(f"   - 匿名jobs (user_id=NULL): {stats['anon_jobs']}")
        print(f"   - 用户jobs (user_id不为NULL): {stats['user_jobs']}")
        print(f"   - 没有device_fingerprint的jobs: {stats['no_fingerprint']}")
        if stats['no_fingerprint'] > 0:
            print(f"   ⚠️  警告: 有 {stats['no_fingerprint']} 个jobs没有device_fingerprint")
    
    conn.close()
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    check_data_isolation()

