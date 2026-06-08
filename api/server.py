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

from ..core.graph import ComputationGraph
from ..core.engine import TrainingEngine
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
                graph_data = json.loads(graph_json)
                graph = ComputationGraph.from_dict(graph_data)
                viz = NetworkVisualizer(graph)
                fig_data = viz.to_json()
                self._json({'visualization': fig_data})
            except Exception as e:
                self._json({'error': str(e)})
        elif path == '/api/graph/layers':
            graph_json = params.get('graph', ['{}'])[0]
            try:
                graph_data = json.loads(graph_json)
                graph = ComputationGraph.from_dict(graph_data)
                self._json({'layers': [n.to_dict() for n in graph.nodes], 'edges': graph.edges})
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
                    graph.add_node(n['id'], n.get('type', 'linear'), n.get('params', {}))
                for e in edges:
                    graph.add_edge(e['from'], e['to'])
                graph.topological_sort()
                self._json({'graph': graph.to_dict(), 'param_count': graph.param_count()})
            except Exception as ex:
                self._json({'error': str(ex)})
        elif parsed.path == '/api/train':
            try:
                from ..models.architectures import ArchitectureRegistry
                arch_name = data.get('architecture', 'mlp')
                arch_params = data.get('params', {})
                model = ArchitectureRegistry.build(arch_name, **arch_params)
                engine = TrainingEngine(model, **data.get('engine', {}))
                result = engine.run()
                self._json(result)
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
