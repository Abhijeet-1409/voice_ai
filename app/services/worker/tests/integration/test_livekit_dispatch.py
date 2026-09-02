import asyncio
import json
import os
import uuid

from livekit import api


ROOM_PREFIX = "worker-playground"


async def main() -> None:
    livekit_url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    agent_name = os.environ["AGENT_NAME"]
    phone_no = os.environ["PHONE_NO"]
    room_timeout= int(os.environ["LIVEKIT_ROOM_TIMEOUT"])

    room_name = f"{ROOM_PREFIX}-{uuid.uuid4().hex[:8]}"

    room_metadata = {
        "stream_sid": f"test-stream-{uuid.uuid4().hex[:8]}",
        "channel": "phone",
        "call_type": "inbound",
        "phone": f"+91{phone_no}",
    }

    async with api.LiveKitAPI(
        url=livekit_url,
        api_key=api_key,
        api_secret=api_secret,
    ) as lkapi:

        room = await lkapi.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=room_timeout,
                metadata=json.dumps(room_metadata),
            )
        )

        print(f"Room created: {room.name}")
        print(f"Room metadata: {room.metadata}")

        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name=agent_name,
            )
        )

        print(f"Dispatch created: {dispatch.id}")
        print(f"Agent: {dispatch.agent_name}")
        print(f"Room: {dispatch.room}")
        print()
        print("Join this room from the LiveKit Playground:")
        print(room_name)


if __name__ == "__main__":
    asyncio.run(main())