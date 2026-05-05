import pandas as pd
from statsbombpy import sb

# La Liga = competition_id 11
# Temporadas mais recentes e completas
season_ids = [90, 42, 4, 1, 2, 27, 26, 25]  # 2020/21 até 2013/14

all_shots = []

for season_id in season_ids:
    matches = sb.matches(competition_id=11, season_id=season_id)
    print(f"Season {season_id}: {len(matches)} partidas")
    for match_id in matches['match_id']:
        events = sb.events(match_id=match_id)
        shots = events[events['type'] == 'Shot']
        all_shots.append(shots)

shots_df = pd.concat(all_shots, ignore_index=True)
print(f"\nTotal de chutes coletados: {len(shots_df)}")
shots_df.to_csv('data/statsbomb_shots_laliga.csv', index=False)
print("Salvo em data/statsbomb_shots_laliga.csv")