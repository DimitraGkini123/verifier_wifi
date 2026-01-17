import socket, json, time

PC_IP = "192.168.68.104"   # ή βάλε την IP του PC αν το τρέξεις από αλλού
PORT = 4242

def send(s, obj):
    s.sendall((json.dumps(obj) + "\n").encode())

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((PC_IP, PORT))
    send(s, {"type":"HELLO","device_id":"pico2w_1"})

    buf = b""
    while True:
        data = s.recv(4096)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            req = json.loads(line.decode())
            print("[FAKE] RX", req)

            # reply with same req_id
            rid = req.get("req_id")

            if req.get("type") == "PING":
                send(s, {"type":"PONG","req_id":rid})
            elif req.get("type") == "GET_METRICS":
                send(s, {"type":"METRICS","req_id":rid,"dC":123,"dL":4,"dP":7,"dE":1})
            elif req.get("type") == "ATTEST_REQUEST" and req.get("mode") == "FULL_HASH_PROVER":
                # dummy reply
                send(s, {"type":"ATTEST_RESPONSE","req_id":rid,"hash_hex":"deadbeef"})
            elif req.get("type") == "ATTEST_REQUEST" and req.get("mode") == "PARTIAL_BLOCKS":
                indices = req.get("indices", [])
                blocks = [{"index":i, "hash_hex":"00"*32} for i in indices]
                send(s, {"type":"ATTEST_RESPONSE","req_id":rid,"blocks":blocks})
            else:
                send(s, {"type":"ERROR","req_id":rid,"reason":"unknown"})
        time.sleep(0.01)
