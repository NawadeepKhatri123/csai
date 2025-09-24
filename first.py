
import pandas as pd
from pyvis.network import Network

# Load the battles dataset
data = pd.read_csv("game-of-thrones-battles.csv")

# Keep only the columns we need
battles_df = data.loc[:, ['name','attacker_king','defender_king','attacker_size','defender_size']]

# Remove rows with missing values
battles_df_cleaned = battles_df.dropna()

# Create the network
net5kings = Network(
    heading="War of the Five Kings",
    bgcolor="#242020",
    font_color="white",
    height="800px",
    width="100%",
    directed=True
)

# Define colors for each king
kingColors = {
    "Joffrey/Tommen Baratheon": "red",
    "Robb Stark": "blue",
    "Stannis Baratheon": "purple",
    "Balon/Euron Greyjoy": "green",
    "Renly Baratheon": "orange",
    "Mance Rayder": "gold"
}

# Add nodes (kings) to the network
kings = set(battles_df_cleaned['attacker_king']).union(set(battles_df_cleaned['defender_king']))
for king in kings:
    net5kings.add_node(
        king,
        label=king,
        shape="dot",
        font={"color": "white"},
        color=kingColors.get(king, "gray")  # fallback if missing
    )

# Count battles between attacker and defender
edges = battles_df_cleaned.groupby(['attacker_king','defender_king']).agg(
    battle_count=('name','count'),
    battle_names=('name', lambda x: ", ".join(x))
).reset_index()

# Add edges with weight (battle count) and title (battle names)
for _, row in edges.iterrows():
    net5kings.add_edge(
        row['attacker_king'],
        row['defender_king'],
        value=row['battle_count'],  # edge thickness
        title=f"{row['battle_count']} battle(s): {row['battle_names']}"
    )

# Scale node size by number of unique enemies attacked
enemies_map = net5kings.get_adj_list()
for node in net5kings.nodes:
    king = node["id"]
    enemies = enemies_map.get(king, set())
    node["value"] = len(enemies) + 1  # bigger = more enemies attacked

# Save the interactive graph to HTML
net5kings.show("Lab1-task1-net5kings.html", notebook=False)
print("Graph saved → Lab1-task1-net5kings.html")
