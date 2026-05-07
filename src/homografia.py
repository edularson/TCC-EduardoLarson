from sports.configs.soccer import SoccerPitchConfiguration
config = SoccerPitchConfiguration()
print(f"Total vertices: {len(config.vertices)}")
print(f"Primeiros 5: {config.vertices[:5]}")
print(f"Últimos 5: {config.vertices[-5:]}")

from sports.configs.soccer import SoccerPitchConfiguration
config = SoccerPitchConfiguration()
for i, v in enumerate(config.vertices):
    print(f"Keypoint {i}: ({v[0]/100:.1f}m, {v[1]/100:.1f}m)")