import requests
from urllib.parse import urlparse


def test_proxy(proxy_url, target_url="https://www.baidu.com"):
    """
    测试代理是否正常工作

    Args:
        proxy_url: 代理地址，格式如 "host:port" 或 "http://host:port"
        target_url: 要测试访问的目标URL
    """
    # 确保代理URL格式正确
    if not proxy_url.startswith(("http://", "https://")):
        proxy_url = f"http://{proxy_url}"

    # 设置代理
    proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }

    print(f"正在测试代理: {proxy_url}")
    print(f"目标网站: {target_url}")
    print("-" * 50)

    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        }

        # 发送请求，设置超时时间
        response = requests.get(
            target_url,
            proxies=proxies,
            headers=headers,
            timeout=10,
            verify=False  # 如果是自签名证书的代理，可能需要关闭验证
        )

        # 输出结果
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {response.elapsed.total_seconds():.2f}秒")

        if response.status_code == 200:
            # 检查是否成功访问到百度
            if "百度一下" in response.text or "baidu" in response.text.lower():
                print("✓ 代理工作正常，成功访问百度")
                # 可以检查响应内容长度
                print(f"响应内容长度: {len(response.text)} 字符")

                # 可选：显示部分响应内容
                print("\n响应内容预览 (前500字符):")
                print(response.text[:500])
            else:
                print("⚠ 代理可以连接，但可能未正确访问到目标网站")
                print("响应内容前200字符:")
                print(response.text[:200])
        else:
            print(f"✗ 代理连接失败，状态码: {response.status_code}")

    except requests.exceptions.ProxyError as e:
        print(f"✗ 代理连接错误: {e}")
        print("可能原因：代理服务器无法连接或拒绝连接")

    except requests.exceptions.ConnectTimeout as e:
        print(f"✗ 连接超时: {e}")
        print("可能原因：代理服务器响应太慢或网络不通")

    except requests.exceptions.Timeout as e:
        print(f"✗ 请求超时: {e}")
        print("可能原因：代理服务器处理请求时间过长")

    except requests.exceptions.SSLError as e:
        print(f"✗ SSL证书错误: {e}")
        print("尝试使用HTTP代理或关闭证书验证")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")

    except Exception as e:
        print(f"✗ 未知错误: {e}")


def main():
    # 你的代理地址
    proxy_address = "g347eb69.natappfree.cc:26799"

    # 可选：也可以测试其他网站
    # target_url = "https://httpbin.org/ip"  # 这个网站会返回你的IP地址

    print("=" * 50)
    print("代理连接测试")
    print("=" * 50)

    # 测试代理
    test_proxy(proxy_address)

    # 可选：测试不使用代理
    print("\n" + "=" * 50)
    print("对比测试：不使用代理")
    print("=" * 50)
    try:
        response = requests.get("https://www.baidu.com", timeout=10)
        print(f"直接访问状态码: {response.status_code}")
        print(f"直接访问响应时间: {response.elapsed.total_seconds():.2f}秒")
    except Exception as e:
        print(f"直接访问失败: {e}")


if __name__ == "__main__":
    # 如果遇到SSL证书警告，可以取消下面的注释
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    main()