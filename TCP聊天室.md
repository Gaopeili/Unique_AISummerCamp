# TCP聊天室

### Server & Clients

* 功能和目标：

  - [x] clients和server可以通信

  - [x] clients进入聊天室需要注册
  - [x] client进入、退出聊天室，其他client会收到系统提示
  - [x] server可以同时和多个clients通信
  - [x] clients之间可以通信

* 实现

  * server使用多线程和多个clients交流

    > 每个client对象由一个线程调控

  * client使用多线程进行全双工通信(收发可以同步)

    > 每个client的收、发由两个线程调控，实现同步

  * server实现了client登录(login)、发送消息(send)、接收消息(recv)

  * client实现了多clients交流

    > client将需要发送的消息发给server，由server发送给其他的clients

* 详细设计

  > * 通过`socket`搭建网络连接
  > * 实现登录：
  >   * server给client发送提示注册信息
  >   * client向server发送注册信息
  >   * server查看是否重复，重复则继续登录
  > * 交流通信：
  >   * 不同client可以通过中转站server获取到其他client的消息
  >   * client的send线程和recv线程维持同步通信
  > * 退出：
  >   * client发送quit
  >   * server接收到quit将该client对应的记录删除并且关闭对应线程

* 代码

  * client代码

    ```python
    from socket import *
    import threading
    import sys
    import signal
    
    flag = True
    
    
    class Client():
        def __init__(self, host, port):
            # ip地址+端口号
            self.address = (host, port)
            # 客户端
            self.client = socket(AF_INET, SOCK_STREAM)
    
            try:
                self.client.connect(self.address)
            except BaseException:
                print("连接失败")
                self.client.close()
    
        # 循环发送数据
        def send(self):
            while True:
                msg = input("~$ ")
                self.client.send(msg.encode('utf-8'))
                if msg == "quit":
                    flag = False
                    break
    
        # 接收数据
        def recv(self):
            while True:
                msg = self.client.recv(1024).decode('utf-8')
    
                print(msg)
                if flag == False:
                    break
    
    
    client = Client("192.168.1.104", 4567)
    
    # 设置昵称
    while True:
        msg = client.client.recv(1024).decode('utf-8')
        print(msg)
        if msg == "请输入昵称:":
            nickname = input()
            client.client.send(nickname.encode('utf-8'))
            msg = client.client.recv(1024).decode('utf-8')
            print(msg)
            if msg == "昵称设置成功\n":
    
                break
    
    # 收发信息
    Thread_send = threading.Thread(target=client.send)
    Thread_recv = threading.Thread(target=client.recv)
    
    Thread_send.start()
    Thread_recv.start()
    ```

  * server代码：

    ```python
    from socket import *
    import threading
    import time
    
    abandon_thread = []
    class Server():
    
        def __init__(self, host, port):
    
            # 地址
            self.address = (host, port)
            # 服务器对象
            self.server = socket(AF_INET, SOCK_STREAM)
            # 端口重用
            self.server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
            # 绑定地址
            self.server.bind(self.address)
            # 开始监听
            self.server.listen(100)
            print("服务器开始监听")
            # 客户端以及对应的Thread对象和昵称  client:[Thread,nickname]
            self.clients = {}
            # 客户端昵称
            self.nickname = []
    
        def run(self):
    
            while True:
    
                if len(abandon_thread) != 0:
                    thread = abandon_thread.pop()
                    thread.join()
    
    
                try:
                    client, addr = self.server.accept()
                except KeyboardInterrupt:
                    self.server.close()
    
                print(str(addr),"连接成功")
                self.clients[client] = {}
                self.login(client)
                # TODO 这里还需要指定target
                Thread = threading.Thread(target=self.activity,args=(client,))
                Thread.start()
    
                # 完善clients列表字典
                self.clients[client]["Thread"] = Thread
    
    
    
    
    
        # 用户活动: 聊天、退出
        def activity(self, client):
            while True:
                # 接受到的msg
                msg = client.recv(1024).decode("utf-8")
                if msg == "quit":
                    self.quit(client)
                    break
                else:
                    self.speak_toall(client, msg)
    
        # 用户登录
        def login(self, client: socket):
    
            # 向用户发送提示消息
            client.send(("请输入昵称:").encode("utf-8"))
            # 用户昵称
            nickname = client.recv(1024).decode('utf-8')
            # 如果当前昵称不存在则可以设置
            if nickname not in self.nickname:
                # 向用户发送提示消息
                client.send(("昵称设置成功\n").encode('utf-8'))
                # 将新昵称加入到昵称列表中
                self.nickname.append(nickname)
                #　完善self.clients列表字典
                self.clients[client]["nickname"] = nickname
    
    
                # 向所有用户发送信息
                for all_client in self.clients.keys():
                    all_client.send(("系统消息"+nickname + "进入了聊天室\n").encode('utf-8'))
            # 如果当前昵称存在则再次注册登录
            else:
                client.send(("当前昵称已经存在，请重新输入\n").encode("utf-8"))
                self.login(client)
    
            return
    
        # 用户发言
        def speak_toall(self, client, msg):
            # 当前用户的昵称
            nickname = self.clients[client]["nickname"]
            # 向所有用户发送消息
            for all_client in self.clients.keys():
                if client != all_client:
                    all_client.send((nickname + ":" + msg+"\n").encode('utf-8'))
    
            return
    
        # 用户退出聊天室
        def quit(self, client:socket):
            nickname = self.clients[client]['nickname']
            for all_client in self.clients.keys():
                if all_client != client:
                    all_client.send(("系统消息"+nickname + "退出了聊天室\n").encode('utf-8'))
                else:
                    all_client.send(("退出").encode('utf-8'))
    
            self.nickname.remove(nickname)
            Thread = self.clients[client]["Thread"]
            abandon_thread.append(Thread)
            del self.clients[client]
            client.close()
    
    
    
    
    
    server = Server("192.168.1.104", 4567)
    server.run()
    
    
    ```

