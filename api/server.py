import json
import os
import io
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from ..core.graph import ComputationGraph, LayerNode, LayerConfig
from ..core.engine import TrainingEngine, TrainingConfig
from ..core.visualizer import NetworkVisualizer


class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/api/status':
            self._json({'status': 'ready', 'matplotlib': HAS_MPL})
        elif path == '/api/architectures':
            from ..models.architectures import ArchitectureRegistry
            self._json({'architectures': ArchitectureRegistry.list()})
        elif path == '/api/graph/visualize':
            graph_json = params.get('graph', ['{}'])[0]
            try:
                graph = ComputationGraph.from_json(graph_json)
                nodes_data = [n.to_dict() for n in graph.nodes.values()]
                viz = NetworkVisualizer()
                layout = viz.compute_layout(nodes_data)
                self._json({'visualization': layout})
            except Exception as e:
                self._json({'error': str(e)})
        elif path == '/api/graph/layers':
            graph_json = params.get('graph', ['{}'])[0]
            try:
                graph = ComputationGraph.from_json(graph_json)
                nodes_data = [n.to_dict() for n in graph.nodes.values()]
                edges = []
                for nid, node in graph.nodes.items():
                    for out_id in node.outputs:
                        edges.append({'from': nid, 'to': out_id})
                self._json({'layers': nodes_data, 'edges': edges})
            except Exception as e:
                self._json({'error': str(e)})
        elif path == '/':
            self._file(os.path.join(os.path.dirname(__file__), '..', 'web', 'index.html'))
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode() if length else '{}'

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json({'error': 'invalid JSON'})
            return

        if parsed.path == '/api/graph/build':
            try:
                graph = ComputationGraph()
                nodes = data.get('nodes', [])
                edges = data.get('edges', [])
                for n in nodes:
                    config = LayerConfig(
                        layer_type=n.get('type', 'linear'),
                        params=n.get('params', {}),
                    )
                    node = LayerNode(config, n['id'])
                    graph.add_node(node)
                for e in edges:
                    graph.connect(e['from'], e['to'])
                order = graph.topological_sort()
                self._json({'nodes': [graph.nodes[nid].to_dict() for nid in order], 'param_count': graph.get_param_count()})
            except Exception as ex:
                self._json({'error': str(ex)})
        elif parsed.path == '/api/train':
            try:
                from ..models.architectures import ArchitectureRegistry
                arch_name = data.get('architecture', 'mlp')
                arch_params = data.get('params', {})
                model = ArchitectureRegistry.build(arch_name, **arch_params)
                engine_cfg = data.get('engine', {})
                config = TrainingConfig(**engine_cfg)
                engine = TrainingEngine(config)
                result = engine.train(model, data.get('train_loader', []), data.get('val_loader', []))
                self._json({'metrics': [vars(m) for m in result]})
            except Exception as ex:
                self._json({'error': str(ex)})
        else:
            self.send_error(404)

    def _json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _file(self, path):
        try:
            with open(path, 'rb') as f:
                self.send_response(200)
                if path.endswith('.html'):
                    self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def run_server(host='0.0.0.0', port=7070):
    server = HTTPServer((host, port), APIHandler)
    print(f"Neural Network Playground API on http://{host}:{port}")
    server.serve_forever()


if __name__ == '__main__':
    run_server()
