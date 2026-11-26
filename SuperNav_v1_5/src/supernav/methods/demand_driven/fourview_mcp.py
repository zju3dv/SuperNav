"""Four-view navigation tools. Only public observations cross this boundary."""
import argparse
import base64
import json
import urllib.error
import urllib.request

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--bridge-port",type=int,default=None)
    parser.add_argument("--transport",choices=("stdio","streamable-http"),default="stdio")
    args=parser.parse_args(argv)
    from mcp.server.fastmcp import FastMCP, Image
    mcp=FastMCP("four-view-demand-navigation",host="127.0.0.1",port=args.port)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,payload=None):
        req=urllib.request.Request("http://127.0.0.1:%s%s"%(args.bridge_port or args.port,path),
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type":"application/json"})
        try:
            with opener.open(req,timeout=300) as response:result=json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(json.loads(exc.read()).get("error","Navigation call rejected")) from None
        if "rgb_jpeg_base64" in result:
            raise ValueError("Front-only bridge requires MCP --views front")
        images=result.pop("images",{})
        content=[json.dumps(result,ensure_ascii=False)]
        for view in ("front","right","back","left"):
            if view in images:
                content.extend([view+" view",Image(data=base64.b64decode(images[view]),format="jpeg")])
        return content

    @mcp.tool()
    def ddn_observe():
        """Read the demand and latest front/right/back/left RGB views. No map, GT or success feedback."""
        return request("/observation")

    @mcp.tool()
    def ddn_step(action:str):
        """Execute forward/backward 0.1m or left/right/look_up/look_down 10deg; return fresh four views."""
        return request("/step",{"action":action})

    @mcp.tool()
    def ddn_local_navigate(observation_id:str,view:str,x:float,y:float,max_replans:int=4):
        """Approach a normalized pixel in a named latest view using front-RGB NoMaD. Side-view alignment consumes physical turns. Local arrival is not task completion."""
        return request("/navigate",dict(observation_id=observation_id,view=view,pixel=[x,y],max_replans=max_replans))

    @mcp.tool()
    def ddn_claim_resource(observation_id:str,view:str,description:str,x:float,y:float):
        """Record a resource arrival claim and its pixel in a latest view. No correctness feedback is returned."""
        return request("/claim",dict(observation_id=observation_id,view=view,description=description,pixel=[x,y]))

    @mcp.tool()
    def ddn_stop():
        """Explicitly STOP and end the episode. Costs one action unit; does not assign success."""
        return request("/stop",{})
    mcp.run(transport=args.transport)

if __name__=="__main__":main()
