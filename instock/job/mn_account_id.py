import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 1. 数据库连接配置
# 请根据实际情况修改以下参数
DB_CONFIG = {
    'host': '192.168.31.192',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'stock_master'
}


class PaginatedChartViewer:
    """分页图表查看器"""

    def __init__(self, df):
        self.df = df
        self.account_ids = df['mn_account_id'].unique()
        self.current_page = 0
        self.accounts_per_page = 9  # 每页显示9个账户（3x3网格）

    def get_page_accounts(self, page):
        """获取指定页的账户ID"""
        start_idx = page * self.accounts_per_page
        end_idx = start_idx + self.accounts_per_page
        return self.account_ids[start_idx:end_idx]

    def get_total_pages(self):
        """获取总页数"""
        return (len(self.account_ids) + self.accounts_per_page - 1) // self.accounts_per_page

    def plot_page(self, page=None):
        """绘制指定页的图表"""
        if page is None:
            page = self.current_page

        if page < 0 or page >= self.get_total_pages():
            print(f"页码 {page + 1} 无效，有效范围: 1-{self.get_total_pages()}")
            return

        self.current_page = page
        page_accounts = self.get_page_accounts(page)

        # 创建图形
        n_accounts = len(page_accounts)
        n_cols = 3
        n_rows = (n_accounts + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        fig.suptitle(f'账户收益走势图 (第 {page + 1}/{self.get_total_pages()} 页)',
                     fontsize=14, fontweight='bold', y=1.02)

        # 如果只有一行，确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_accounts == 1:
            axes = np.array([[axes]])

        # 绘制每个账户的子图
        for idx, account_id in enumerate(page_accounts):
            row = idx // n_cols
            col = idx % n_cols

            account_data = self.df[self.df['mn_account_id'] == account_id]

            if len(account_data) > 0:
                ax = axes[row, col] if n_rows > 1 else axes[0, col]

                # 绘制收益走势线
                ax.plot(account_data['date'],
                        account_data['total_earnings'],
                        'b-',
                        linewidth=2,
                        alpha=0.7)

                # 添加数据点标记
                ax.scatter(account_data['date'],
                           account_data['total_earnings'],
                           c='blue',
                           s=30,
                           alpha=0.6,
                           zorder=5)

                # 标记最高点和最低点
                if len(account_data) > 1:
                    max_idx = account_data['total_earnings'].idxmax()
                    min_idx = account_data['total_earnings'].idxmin()

                    ax.scatter(account_data.loc[max_idx, 'date'],
                               account_data.loc[max_idx, 'total_earnings'],
                               color='green',
                               s=100,
                               zorder=6,
                               label='最高点',
                               marker='^')
                    ax.scatter(account_data.loc[min_idx, 'date'],
                               account_data.loc[min_idx, 'total_earnings'],
                               color='red',
                               s=100,
                               zorder=6,
                               label='最低点',
                               marker='v')

                # 设置标题和标签
                title = f'账户: {account_id}\n数据点: {len(account_data)}'
                ax.set_title(title, fontsize=11, fontweight='bold')

                # 计算并显示统计信息
                stats_text = f"""平均值: {account_data['total_earnings'].mean():.2f}
最大值: {account_data['total_earnings'].max():.2f}
最小值: {account_data['total_earnings'].min():.2f}"""

                # 将统计信息添加到图表
                ax.text(0.02, 0.98, stats_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                ax.legend(loc='upper left', fontsize=8)

        # 隐藏多余的子图
        for idx in range(len(page_accounts), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
            else:
                axes[0, col].axis('off')

        plt.tight_layout()
        plt.show()

        # 显示当前页的账户信息
        self.show_page_info(page, page_accounts)

    def show_page_info(self, page, page_accounts):
        """显示当前页的账户信息"""
        print(f"\n{'=' * 60}")
        print(f"第 {page + 1}/{self.get_total_pages()} 页")
        print(f"本页账户: {', '.join(page_accounts)}")
        print(f"{'=' * 60}")

        # 显示本页账户的简要统计
        for account_id in page_accounts:
            account_data = self.df[self.df['mn_account_id'] == account_id]
            if len(account_data) > 0:
                print(f"\n账户 {account_id}:")
                print(f"  数据点数量: {len(account_data)}")
                print(f"  日期范围: {account_data['date'].min().date()} 至 {account_data['date'].max().date()}")
                print(
                    f"  收益范围: {account_data['total_earnings'].min():.2f} 至 {account_data['total_earnings'].max():.2f}")
                print(f"  平均收益: {account_data['total_earnings'].mean():.2f}")

    def interactive_navigation(self):
        """交互式导航"""
        while True:
            print(f"\n{'=' * 60}")
            print(f"当前在第 {self.current_page + 1}/{self.get_total_pages()} 页")
            print("导航选项:")
            print("  n: 下一页")
            print("  p: 上一页")
            print("  g <页码>: 跳转到指定页码 (如: g 3)")
            print("  s <账户ID>: 搜索特定账户 (如: s ACCT001)")
            print("  f: 返回第一页")
            print("  l: 跳转到最后一页")
            print("  q: 退出查看")
            print(f"{'=' * 60}")

            command = input("请输入命令: ").strip().lower()

            if command == 'q':
                print("退出分页查看模式")
                break
            elif command == 'n':
                if self.current_page < self.get_total_pages() - 1:
                    self.current_page += 1
                    self.plot_page(self.current_page)
                else:
                    print("已经是最后一页了！")
            elif command == 'p':
                if self.current_page > 0:
                    self.current_page -= 1
                    self.plot_page(self.current_page)
                else:
                    print("已经是第一页了！")
            elif command == 'f':
                self.current_page = 0
                self.plot_page(self.current_page)
            elif command == 'l':
                self.current_page = self.get_total_pages() - 1
                self.plot_page(self.current_page)
            elif command.startswith('g '):
                try:
                    page_num = int(command.split()[1]) - 1
                    if 0 <= page_num < self.get_total_pages():
                        self.current_page = page_num
                        self.plot_page(self.current_page)
                    else:
                        print(f"页码 {page_num + 1} 无效，有效范围: 1-{self.get_total_pages()}")
                except (ValueError, IndexError):
                    print("无效的命令格式，请使用: g <页码>")
            elif command.startswith('s '):
                try:
                    account_id = command.split()[1]
                    if account_id in self.account_ids:
                        # 找到账户所在的页码
                        account_index = np.where(self.account_ids == account_id)[0][0]
                        target_page = account_index // self.accounts_per_page
                        self.current_page = target_page
                        print(f"账户 {account_id} 在第 {target_page + 1} 页")
                        self.plot_page(target_page)
                    else:
                        print(f"未找到账户: {account_id}")
                except (IndexError, ValueError):
                    print("无效的命令格式，请使用: s <账户ID>")
            else:
                print("未知命令，请重新输入")


def fetch_data_from_mysql():
    """
    从MySQL数据库获取数据
    """
    try:
        # 创建数据库连接引擎
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        # SQL查询语句
        query = """
        SELECT 
            date,
            mn_account_id,
            CAST(total_earnings AS DECIMAL(15,2)) as total_earnings
        FROM investment_portfolio_snapshot
        WHERE mn_account_id IS NOT NULL 
            AND total_earnings IS NOT NULL 
            AND total_earnings != ''
        ORDER BY date, mn_account_id
        """

        # 读取数据
        df = pd.read_sql(query, engine, parse_dates=['date'])

        print(f"成功读取 {len(df)} 条记录")
        print(f"账户ID数量: {df['mn_account_id'].nunique()}")
        print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")

        return df

    except Exception as e:
        print(f"数据库连接或查询出错: {e}")
        return None


def prepare_data(df):
    """
    准备数据，处理数据格式
    """
    # 确保total_earnings是数值类型
    df['total_earnings'] = pd.to_numeric(df['total_earnings'], errors='coerce')

    # 删除NaN值
    df = df.dropna(subset=['date', 'mn_account_id', 'total_earnings'])

    # 按日期和账户ID排序
    df = df.sort_values(['mn_account_id', 'date'])

    # 获取账户ID列表
    account_ids = df['mn_account_id'].unique()

    print(f"\n数据统计:")
    print(f"有效记录数: {len(df)}")
    print(f"有效账户ID数: {len(account_ids)}")

    return df, account_ids


def plot_overall_trends(df):
    """
    绘制整体趋势图表
    """
    print("\n正在生成整体趋势分析...")

    # 创建一个大图，包含多个子图
    fig = plt.figure(figsize=(18, 12))

    # 1. 所有账户整体收益趋势（汇总）
    ax1 = plt.subplot(2, 3, 1)

    # 计算每日总收益
    daily_total = df.groupby('date')['total_earnings'].sum()
    # 计算每日平均收益
    daily_avg = df.groupby('date')['total_earnings'].mean()
    # 计算每日账户数量
    daily_accounts = df.groupby('date')['mn_account_id'].nunique()

    # 绘制总收益
    color = 'tab:blue'
    ax1.plot(daily_total.index, daily_total.values, color=color, linewidth=2, label='每日总收益')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('总收益', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('整体收益趋势（总收益）', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 创建第二个y轴用于显示账户数量
    ax1b = ax1.twinx()
    color = 'tab:red'
    ax1b.plot(daily_accounts.index, daily_accounts.values, color=color, linewidth=1.5, alpha=0.7, label='活跃账户数')
    ax1b.set_ylabel('活跃账户数', color=color)
    ax1b.tick_params(axis='y', labelcolor=color)
    ax1b.legend(loc='upper right')

    # 2. 每日平均收益趋势
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(daily_avg.index, daily_avg.values, 'g-', linewidth=2, label='平均收益')

    # 添加移动平均线（7天）
    if len(daily_avg) > 7:
        moving_avg = daily_avg.rolling(window=7).mean()
        ax2.plot(moving_avg.index, moving_avg.values, 'r--', linewidth=1.5, alpha=0.7, label='7日移动平均')

    ax2.set_xlabel('日期')
    ax2.set_ylabel('平均收益')
    ax2.set_title('每日平均收益趋势', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 收益分布热力图（按账户）
    ax3 = plt.subplot(2, 3, 3)

    # 选择数据最多的前20个账户
    top_accounts = df['mn_account_id'].value_counts().head(20).index
    top_data = df[df['mn_account_id'].isin(top_accounts)]

    # 创建透视表
    pivot_table = top_data.pivot_table(index='mn_account_id', columns='date', values='total_earnings')

    # 绘制热力图
    im = ax3.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax3.set_xlabel('日期（索引）')
    ax3.set_ylabel('账户ID')
    ax3.set_title('前20个账户收益热力图', fontsize=12, fontweight='bold')

    # 设置y轴标签（账户ID）
    ax3.set_yticks(range(len(pivot_table.index)))
    ax3.set_yticklabels(pivot_table.index)

    # 添加颜色条
    plt.colorbar(im, ax=ax3, label='收益值')

    # 4. 收益分布箱线图
    ax4 = plt.subplot(2, 3, 4)

    # 按日期分组，获取每日收益分布
    box_data = []
    box_labels = []

    # 选择一些关键日期（每隔N天）
    unique_dates = df['date'].unique()
    if len(unique_dates) > 10:
        # 如果日期太多，选择等间距的10个日期
        step = len(unique_dates) // 10
        selected_dates = unique_dates[::step]
    else:
        selected_dates = unique_dates

    for date in selected_dates:
        daily_earnings = df[df['date'] == date]['total_earnings']
        if len(daily_earnings) > 0:
            box_data.append(daily_earnings.values)
            box_labels.append(date.strftime('%m-%d'))

    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)

    # 设置箱线图颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors * (len(box_data) // len(colors) + 1)):
        patch.set_facecolor(color)

    ax4.set_xlabel('日期')
    ax4.set_ylabel('收益分布')
    ax4.set_title('收益分布箱线图（按日期）', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)

    # 5. 收益排名变化（前10个账户）
    ax5 = plt.subplot(2, 3, 5)

    # 获取每个日期收益排名前10的账户
    top_n = 10
    top_accounts_over_time = []

    for date in sorted(df['date'].unique()):
        date_data = df[df['date'] == date]
        if len(date_data) >= top_n:
            top_on_date = date_data.nlargest(top_n, 'total_earnings')
            top_accounts_over_time.append(top_on_date['mn_account_id'].tolist())

    # 计算每个账户出现在前10名的次数
    from collections import Counter
    all_top_accounts = [account for sublist in top_accounts_over_time for account in sublist]
    top_counter = Counter(all_top_accounts)

    # 获取出现次数最多的前15个账户
    top_15_accounts = [account for account, _ in top_counter.most_common(15)]

    # 计算每个账户在每个日期的排名
    rank_data = {account: [] for account in top_15_accounts}

    for date in sorted(df['date'].unique()):
        date_data = df[df['date'] == date]
        if len(date_data) > 0:
            # 计算当日排名
            date_data = date_data.sort_values('total_earnings', ascending=False)
            date_data['rank'] = range(1, len(date_data) + 1)

            for account in top_15_accounts:
                if account in date_data['mn_account_id'].values:
                    rank = date_data[date_data['mn_account_id'] == account]['rank'].iloc[0]
                    # 转换为排名分数（排名越高，分数越低）
                    rank_score = 1 / rank if rank <= 10 else 0
                else:
                    rank_score = 0
                rank_data[account].append(rank_score)

    # 绘制堆叠面积图
    dates_list = sorted(df['date'].unique())
    dates_formatted = [d.strftime('%m-%d') for d in dates_list]

    bottom = np.zeros(len(dates_list))
    for account in top_15_accounts[:8]:  # 只显示前8个避免过于拥挤
        scores = rank_data[account]
        ax5.fill_between(dates_formatted, bottom, bottom + scores, alpha=0.6, label=account)
        bottom += scores

    ax5.set_xlabel('日期')
    ax5.set_ylabel('排名分数（1/排名）')
    ax5.set_title('前10名账户排名变化', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

    # 6. 收益变化率分布
    ax6 = plt.subplot(2, 3, 6)

    # 计算每个账户的收益变化率
    change_rates = []

    for account_id in df['mn_account_id'].unique():
        account_data = df[df['mn_account_id'] == account_id].sort_values('date')
        if len(account_data) > 1:
            # 计算收益率变化
            account_data['earnings_change'] = account_data['total_earnings'].pct_change() * 100
            valid_changes = account_data['earnings_change'].dropna()
            change_rates.extend(valid_changes.tolist())

    # 绘制直方图
    if change_rates:
        ax6.hist(change_rates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax6.axvline(x=np.mean(change_rates), color='red', linestyle='--', linewidth=2,
                    label=f'均值: {np.mean(change_rates):.2f}%')
        ax6.axvline(x=np.median(change_rates), color='green', linestyle='--', linewidth=2,
                    label=f'中位数: {np.median(change_rates):.2f}%')

        ax6.set_xlabel('收益变化率 (%)')
        ax6.set_ylabel('频次')
        ax6.set_title('收益变化率分布', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, '无足够数据计算变化率',
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('收益变化率分布', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.suptitle('整体收益趋势综合分析', fontsize=16, fontweight='bold', y=1.02)
    plt.show()

    # 打印整体统计信息
    print(f"\n整体统计信息:")
    print(f"数据总天数: {len(daily_total)}")
    print(f"总收益范围: {daily_total.min():,.2f} 到 {daily_total.max():,.2f}")
    print(f"平均每日总收益: {daily_total.mean():,.2f}")
    print(f"平均每日账户数: {daily_accounts.mean():.1f}")
    print(f"平均每日收益/账户: {daily_avg.mean():,.2f}")


def plot_recent_5days_trend(df):
    """
    绘制最近5天的趋势图表
    """
    print("\n正在生成最近5天趋势分析...")

    # 获取最近的日期
    latest_date = df['date'].max()
    five_days_ago = latest_date - timedelta(days=4)  # 包括今天共5天

    # 筛选最近5天的数据
    recent_data = df[df['date'] >= five_days_ago].copy()

    if len(recent_data) == 0:
        print("最近5天没有数据")
        return

    print(f"最近5天数据范围: {recent_data['date'].min().date()} 到 {recent_data['date'].max().date()}")
    print(f"最近5天活跃账户数: {recent_data['mn_account_id'].nunique()}")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 最近5天每日总收益和平均收益
    ax1 = axes[0, 0]

    # 按日期分组计算
    daily_summary = recent_data.groupby('date').agg({
        'total_earnings': ['sum', 'mean', 'count'],
        'mn_account_id': 'nunique'
    }).round(2)

    daily_summary.columns = ['total', 'avg', 'records', 'accounts']

    # 绘制柱状图
    dates_str = [d.strftime('%m-%d') for d in daily_summary.index]
    x = np.arange(len(dates_str))
    width = 0.35

    # 总收益柱状图
    bars1 = ax1.bar(x - width / 2, daily_summary['total'], width, label='总收益', alpha=0.8, color='steelblue')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('总收益', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # 添加总收益数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:,.0f}',
                 ha='center', va='bottom', fontsize=9)

    # 创建第二个y轴用于平均收益
    ax1b = ax1.twinx()
    bars2 = ax1b.bar(x + width / 2, daily_summary['avg'], width, label='平均收益', alpha=0.8, color='orange')
    ax1b.set_ylabel('平均收益', color='orange')
    ax1b.tick_params(axis='y', labelcolor='orange')

    # 添加平均收益数值标签
    for bar in bars2:
        height = bar.get_height()
        ax1b.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{height:.0f}',
                  ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dates_str)
    ax1.set_title('最近5天收益对比', fontsize=12, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 添加账户数量信息
    for i, (date, row) in enumerate(daily_summary.iterrows()):
        ax1.text(i, row['total'] * 0.05, f"账户: {int(row['accounts'])}",
                 ha='center', va='bottom', fontsize=8, color='darkblue')

    # 2. 最近5天账户收益变化热力图（前20个账户）
    ax2 = axes[0, 1]

    # 获取最近5天有数据的账户
    recent_accounts = recent_data['mn_account_id'].value_counts().index

    # 创建透视表
    pivot_data = recent_data.pivot_table(index='mn_account_id', columns='date', values='total_earnings')

    # 按最近一天的数据排序
    if latest_date in pivot_data.columns:
        pivot_data = pivot_data.sort_values(by=latest_date, ascending=False)

    # 只取前20个账户
    top_pivot = pivot_data.head(20)

    # 计算相对于第一天的变化百分比
    if len(top_pivot.columns) > 1:
        first_date = top_pivot.columns[0]
        change_data = top_pivot.copy()
        for col in top_pivot.columns[1:]:
            change_data[col] = (top_pivot[col] - top_pivot[first_date]) / top_pivot[first_date] * 100

        # 绘制热力图
        im = ax2.imshow(change_data.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('账户ID')
        ax2.set_title('前20账户收益变化率热力图（%）', fontsize=12, fontweight='bold')

        # 设置坐标轴标签
        ax2.set_xticks(range(len(change_data.columns)))
        ax2.set_xticklabels([d.strftime('%m-%d') for d in change_data.columns])
        ax2.set_yticks(range(len(change_data.index)))
        ax2.set_yticklabels(change_data.index)

        # 添加颜色条
        plt.colorbar(im, ax=ax2, label='变化率 (%)')
    else:
        ax2.text(0.5, 0.5, '数据不足，需要至少2天数据',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('前20账户收益变化率热力图', fontsize=12, fontweight='bold')

    # 3. 最近5天收益增长最快和最慢的账户
    ax3 = axes[1, 0]

    if len(recent_data['date'].unique()) >= 2:
        # 计算每个账户从第一天到最后一天的收益变化
        first_day = recent_data['date'].min()
        last_day = recent_data['date'].max()

        first_day_data = recent_data[recent_data['date'] == first_day]
        last_day_data = recent_data[recent_data['date'] == last_day]

        # 合并数据
        comparison = pd.merge(
            first_day_data[['mn_account_id', 'total_earnings']],
            last_day_data[['mn_account_id', 'total_earnings']],
            on='mn_account_id',
            suffixes=('_first', '_last')
        )

        # 计算变化金额和变化率
        comparison['change_amount'] = comparison['total_earnings_last'] - comparison['total_earnings_first']
        comparison['change_rate'] = (comparison['change_amount'] / comparison['total_earnings_first'].abs()) * 100

        # 获取增长最快和最慢的10个账户
        top_gainers = comparison.nlargest(10, 'change_amount')
        top_losers = comparison.nsmallest(10, 'change_amount')

        # 绘制柱状图
        x_gainers = np.arange(len(top_gainers))
        x_losers = np.arange(len(top_losers))

        ax3.bar(x_gainers, top_gainers['change_amount'], alpha=0.7, color='green', label='增长最多')
        ax3.bar(x_losers + len(top_gainers) + 1, top_losers['change_amount'], alpha=0.7, color='red', label='减少最多')

        ax3.set_xlabel('账户排名')
        ax3.set_ylabel('收益变化金额')
        ax3.set_title('最近5天收益变化最大的账户', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 设置x轴标签
        gainer_labels = [f"{id[:8]}..." for id in top_gainers['mn_account_id']]
        loser_labels = [f"{id[:8]}..." for id in top_losers['mn_account_id']]
        all_labels = gainer_labels + [''] + loser_labels
        ax3.set_xticks(range(len(all_labels)))
        ax3.set_xticklabels(all_labels, rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, '需要至少2天数据来计算变化',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('最近5天收益变化最大的账户', fontsize=12, fontweight='bold')

    # 4. 最近5天每日账户收益分布
    ax4 = axes[1, 1]

    # 准备小提琴图数据
    violin_data = []
    violin_labels = []

    for date in sorted(recent_data['date'].unique()):
        daily_earnings = recent_data[recent_data['date'] == date]['total_earnings']
        if len(daily_earnings) > 0:
            violin_data.append(daily_earnings.values)
            violin_labels.append(date.strftime('%m-%d'))

    # 绘制小提琴图
    parts = ax4.violinplot(violin_data, showmeans=True, showmedians=True)

    # 设置小提琴图颜色
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax4.set_xlabel('日期')
    ax4.set_ylabel('收益分布')
    ax4.set_title('最近5天每日收益分布', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(1, len(violin_labels) + 1))
    ax4.set_xticklabels(violin_labels)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.suptitle(f'最近5天收益趋势分析 ({five_days_ago.date()} 至 {latest_date.date()})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()

    # 打印最近5天统计信息
    print(f"\n最近5天详细统计:")
    for date in sorted(recent_data['date'].unique()):
        day_data = recent_data[recent_data['date'] == date]
        print(f"\n{date.date()}:")
        print(f"  活跃账户: {day_data['mn_account_id'].nunique()}")
        print(f"  总收益: {day_data['total_earnings'].sum():,.2f}")
        print(f"  平均收益: {day_data['total_earnings'].mean():,.2f}")
        print(f"  中位数收益: {day_data['total_earnings'].median():,.2f}")
        if len(day_data) > 1:
            print(f"  收益标准差: {day_data['total_earnings'].std():,.2f}")


def main():
    """
    主函数
    """
    print("开始从MySQL数据库读取数据...")

    # 1. 从数据库读取数据
    df = fetch_data_from_mysql()

    if df is None or df.empty:
        print("没有读取到数据或数据为空")
        return

    # 2. 准备数据
    df, account_ids = prepare_data(df)

    print(f"\n数据分析选项:")
    print("1. 查看整体趋势分析")
    print("2. 查看最近5天趋势")
    print("3. 分页查看所有账户详细图表")
    print("4. 查看特定账户")

    while True:
        choice = input("\n请选择要查看的分析类型 (1-4, 输入q退出): ").strip()

        if choice == 'q':
            print("程序结束")
            break

        if choice == '1':
            # 查看整体趋势
            plot_overall_trends(df)

        elif choice == '2':
            # 查看最近5天趋势
            plot_recent_5days_trend(df)

        elif choice == '3':
            # 分页查看所有账户
            if len(account_ids) > 0:
                viewer = PaginatedChartViewer(df)
                print(f"\n共有 {len(account_ids)} 个账户，分为 {viewer.get_total_pages()} 页")
                print(f"每页显示 {viewer.accounts_per_page} 个账户")

                # 显示第一页
                viewer.plot_page(0)

                # 进入交互式导航
                viewer.interactive_navigation()
            else:
                print("没有找到有效的账户数据")

        elif choice == '4':
            # 查看特定账户
            account_to_view = input("请输入要查看的账户ID: ").strip()
            if account_to_view in account_ids:
                account_data = df[df['mn_account_id'] == account_to_view]

                # 绘制特定账户的详细图表
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # 左图：详细走势
                ax1 = axes[0]
                ax1.plot(account_data['date'], account_data['total_earnings'], 'b-o', linewidth=2)
                ax1.set_xlabel('日期')
                ax1.set_ylabel('总收益')
                ax1.set_title(f'账户 {account_to_view} 收益走势', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)

                # 添加数据点标签
                for i, row in account_data.iterrows():
                    ax1.annotate(f'{row["total_earnings"]:.2f}',
                                 (row['date'], row['total_earnings']),
                                 textcoords="offset points",
                                 xytext=(0, 10),
                                 ha='center',
                                 fontsize=8)

                # 右图：统计信息
                ax2 = axes[1]
                ax2.axis('off')

                # 计算统计信息
                stats_text = f"""账户ID: {account_to_view}

数据统计:
记录数量: {len(account_data)}
日期范围: {account_data['date'].min().date()} 至 {account_data['date'].max().date()}

收益统计:
最高收益: {account_data['total_earnings'].max():.2f}
最低收益: {account_data['total_earnings'].min():.2f}
平均收益: {account_data['total_earnings'].mean():.2f}
中位数收益: {account_data['total_earnings'].median():.2f}
收益标准差: {account_data['total_earnings'].std():.2f}

变化分析:"""

                # 添加收益变化信息
                if len(account_data) > 1:
                    account_data_sorted = account_data.sort_values('date')
                    first_earnings = account_data_sorted['total_earnings'].iloc[0]
                    last_earnings = account_data_sorted['total_earnings'].iloc[-1]
                    total_change = last_earnings - first_earnings
                    change_rate = (total_change / first_earnings) * 100 if first_earnings != 0 else 0

                    stats_text += f"""
总变化: {total_change:+.2f} ({change_rate:+.2f}%)
日均变化: {total_change / len(account_data):+.2f}"""

                ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.suptitle(f'账户 {account_to_view} 详细分析', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
            else:
                print(f"未找到账户: {account_to_view}")

        else:
            print("无效选择，请输入1-4或q退出")


if __name__ == "__main__":
    # 注意：需要先安装必要的库
    # pip install pandas matplotlib sqlalchemy pymysql

    main()