import * as express from 'express';
import * as bodyParser from 'body-parser';
import { getBlockchain, generateNextBlock } from './blockchain';
import { getSockets, connectToPeers, initP2PServer } from './p2p';

const initHttpServer = (httpPort: number) => {
    const app = express();
    app.use(bodyParser.json());

    app.get('/blocks', (req, res) => {
        res.send(getBlockchain());
    });

    app.post('/mineBlock', (req, res) => {
        const newBlock = generateNextBlock(req.body.data || 'empty block data');
        res.send(newBlock);
    });

    app.get('/peers', (req, res) => {
        res.send(getSockets().map((s: any) => s._socket.remoteAddress + ':' + s._socket.remotePort));
    });

    app.post('/addPeer', (req, res) => {
        connectToPeers(req.body.peer);
        res.send();
    });

    app.listen(httpPort, () => {
        console.log('Listening http on port: ' + httpPort);
    });
};

export { initHttpServer };