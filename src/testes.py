from statsbombpy import sb
comps = sb.competitions()
print(comps[comps['competition_id'].isin([9, 16])][
    ['competition_id', 'season_id', 'competition_name', 'season_name']
].to_string())