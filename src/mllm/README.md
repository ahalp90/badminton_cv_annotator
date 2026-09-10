# Quickstart
1. Build container with: `apptainer build --fakeroot container.sif container.def`
2. run container by running `run_container.sh`

# Decisions Explained
## Model
The model used is InternVideo3-8B-Instruct (https://huggingface.co/yanziang/InternVideo3-8B-Instruct). In most cases it seems to be an improvement over intern video 2.5, which was originally proposed, so it will be used instead.


# Preliminary Results
## First attempt
This is the output I got with the simplest most basic setup of internvideo3 on a 1 second video using the prompt "Please describe this video in detail.":
    The video captures a moment from a professional badminton match, likely part of the HSBC World Tour Finals in Bangkok, as indicated by the branding on the court and surrounding banners. The court is a standard green playing surface with white lines, surrounded by a red floor area. The players are positioned on opposite sides of the net.

    In the foreground, a player in a red sleeveless top and black shorts is positioned near the center of the court, executing a forehand shot. His body is turned sideways, and his racket is extended forward as he makes contact with the shuttlecock. His stance suggests he is moving laterally to reach the shuttlecock.

    In the background, the opponent, wearing a dark blue shirt and black shorts, is positioned near the back of the court. He is preparing to respond to the shot, with his racket raised and his body slightly crouched, indicating he is anticipating the shuttlecock's trajectory.

    The scoreboard at the top left of the screen shows the players' names: "LEE Z.J." and "GINTING," with scores of 1 and 4 respectively. The match appears to be in the first game, as indicated by the "1" next to the player names.

    The surrounding environment includes various sponsor logos, such as YONEX, HSBC, and 188BET, displayed prominently on the walls and banners. There are also officials and camera operators positioned around the court, capturing the match from different angles. The overall atmosphere is that of a high-stakes, professional badminton tournament.
