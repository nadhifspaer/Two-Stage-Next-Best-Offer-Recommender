# opens one Streamlit session over the websocket, runs the script, reports exceptions, then measures server RSS
import asyncio, sys, json, psutil
from tornado.websocket import websocket_connect
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

port = int(sys.argv[1]); label = sys.argv[2]

def server_pid(port):
    for c in psutil.net_connections(kind="tcp"):
        if c.status == "LISTEN" and c.laddr.port == port:
            return c.pid

async def main():
    ws = await websocket_connect(f"ws://localhost:{port}/_stcore/stream")
    msg = BackMsg(); msg.rerun_script.query_string = ""
    await ws.write_message(msg.SerializeToString(), binary=True)
    texts, exc, finished = [], [], None
    while finished is None:
        data = await asyncio.wait_for(ws.read_message(), 120)
        fm = ForwardMsg(); fm.ParseFromString(data)
        kind = fm.WhichOneof("type")
        if kind == "script_finished":
            finished = fm.script_finished
        elif kind == "delta":
            d = fm.delta
            if d.WhichOneof("type") == "new_element":
                e = d.new_element; k = e.WhichOneof("type")
                if k == "exception": exc.append(e.exception.message)
                if k == "metric": texts.append(f"metric {e.metric.label}={e.metric.body}")
                if k == "arrow_data_frame": texts.append("dataframe")
                if k == "alert": texts.append("alert " + e.alert.body[:110])
    await asyncio.sleep(3)
    out = {"mode": label, "script_finished": str(finished), "exceptions": exc, "elements": texts}
    if "--no-rss" not in sys.argv:
        pid = server_pid(port); p = psutil.Process(pid)
        out.update(server_pid=pid, rss_mb=round(p.memory_info().rss/1e6,1))
    print(json.dumps(out, indent=1))
    ws.close()
asyncio.run(main())
