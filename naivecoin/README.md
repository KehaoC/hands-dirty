# Naivecoin

一个使用TypeScript实现的简单区块链项目。

## 功能

- 基本的区块链结构
- 工作量证明 (PoW) 挖矿
- P2P网络通信
- HTTP API接口

## 安装

```bash
npm install
```

## 编译

```bash
npm run build
```

## 运行

```bash
npm start
```

或者编译并运行：

```bash
npm run dev
```

## API接口

- GET `/blocks` - 获取所有区块
- POST `/mineBlock` - 挖掘新区块
- GET `/peers` - 获取所有对等节点
- POST `/addPeer` - 添加新的对等节点

## P2P网络

项目使用WebSocket实现P2P网络通信，支持节点间的区块链同步。