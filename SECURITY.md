# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Neural Network Playground, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email the maintainer at **0mayankbasena@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
3. You should receive a response within 48 hours.
4. We will work with you to understand and address the issue before any public disclosure.

## Security Considerations

- This project trains and runs neural network models locally. No data is sent to external servers by default.
- The API server (`api/server.py`) binds to `0.0.0.0` by default. Restrict to `127.0.0.1` in production.
- Imported model files (ONNX, TorchScript) should be treated as untrusted input.
- The web UI does not execute server-side code, but be cautious with user-supplied network configurations.
