"""generate_data.py — 90 films · 500 users · archetype-biased ratings"""
import pandas as pd, numpy as np, os
np.random.seed(42)

MOVIES = [
    # (id, title, genres, year, director, imdb, mood_tags, runtime, cast, tagline, language)
    (1,"The Dark Knight","Action|Crime|Drama",2008,"Christopher Nolan",9.0,"dark|intense|gripping",152,"Christian Bale, Heath Ledger","Why so serious?","en"),
    (2,"Mad Max: Fury Road","Action|Adventure|Sci-Fi",2015,"George Miller",8.1,"intense|adrenaline|epic",120,"Tom Hardy, Charlize Theron","What a lovely day.","en"),
    (3,"John Wick","Action|Crime|Thriller",2014,"Chad Stahelski",7.4,"intense|adrenaline|stylish",101,"Keanu Reeves","He killed my dog.","en"),
    (4,"Avengers: Endgame","Action|Adventure|Drama",2019,"Russo Brothers",8.4,"epic|emotional|fun",181,"Robert Downey Jr., Chris Evans","Whatever it takes.","en"),
    (5,"Top Gun: Maverick","Action|Drama",2022,"Joseph Kosinski",8.3,"adrenaline|epic|feel-good",131,"Tom Cruise, Miles Teller","Feel the need.","en"),
    (6,"The Raid","Action|Crime|Thriller",2011,"Gareth Evans",7.6,"intense|adrenaline",96,"Iko Uwais","One building. No way out.","id"),
    (7,"Baby Driver","Action|Crime|Music",2017,"Edgar Wright",7.6,"stylish|fun|adrenaline",113,"Ansel Elgort","All he needs is the right playlist.","en"),
    (8,"Mission Impossible Fallout","Action|Adventure|Thriller",2018,"Christopher McQuarrie",7.7,"adrenaline|intense|fun",147,"Tom Cruise","Some missions are not a choice.","en"),
    (9,"Inception","Action|Adventure|Sci-Fi",2010,"Christopher Nolan",8.8,"mind-bending|intense|epic",148,"Leonardo DiCaprio","Your mind is the scene of the crime.","en"),
    (10,"Interstellar","Adventure|Drama|Sci-Fi",2014,"Christopher Nolan",8.7,"emotional|epic|mind-bending",169,"Matthew McConaughey","Mankind was born on Earth.","en"),
    (11,"The Matrix","Action|Sci-Fi",1999,"Wachowskis",8.7,"mind-bending|stylish|intense",136,"Keanu Reeves, Laurence Fishburne","Free your mind.","en"),
    (12,"Blade Runner 2049","Drama|Mystery|Sci-Fi",2017,"Denis Villeneuve",8.0,"dark|atmospheric|mind-bending",164,"Ryan Gosling","The key to the future.","en"),
    (13,"Arrival","Drama|Mystery|Sci-Fi",2016,"Denis Villeneuve",7.9,"mind-bending|emotional|atmospheric",116,"Amy Adams","Why are they here?","en"),
    (14,"Ex Machina","Drama|Sci-Fi|Thriller",2014,"Alex Garland",7.7,"dark|mind-bending|atmospheric",108,"Domhnall Gleeson","Thinking makes it so.","en"),
    (15,"Dune","Adventure|Drama|Sci-Fi",2021,"Denis Villeneuve",8.0,"epic|atmospheric|mind-bending",155,"Timothee Chalamet","Beyond fear, destiny awaits.","en"),
    (16,"Everything Everywhere All at Once","Action|Adventure|Comedy",2022,"Daniels",7.8,"mind-bending|emotional|fun",139,"Michelle Yeoh","The universe is saved.","en"),
    (17,"Annihilation","Adventure|Drama|Sci-Fi",2018,"Alex Garland",6.8,"dark|mind-bending|atmospheric",115,"Natalie Portman","Fear what's inside.","en"),
    (18,"Gravity","Adventure|Drama|Sci-Fi",2013,"Alfonso Cuaron",7.7,"intense|adrenaline|epic",91,"Sandra Bullock","Don't let go.","en"),
    (19,"The Shawshank Redemption","Drama",1994,"Frank Darabont",9.3,"emotional|inspiring|gripping",142,"Tim Robbins, Morgan Freeman","Hope can set you free.","en"),
    (20,"Forrest Gump","Drama|Romance",1994,"Robert Zemeckis",8.8,"emotional|feel-good|inspiring",142,"Tom Hanks","Life is like a box of chocolates.","en"),
    (21,"The Godfather","Crime|Drama",1972,"Francis Ford Coppola",9.2,"dark|gripping|intense",175,"Marlon Brando, Al Pacino","An offer you can't refuse.","en"),
    (22,"Fight Club","Drama|Mystery|Thriller",1999,"David Fincher",8.8,"dark|mind-bending|intense",139,"Brad Pitt, Edward Norton","Mischief. Mayhem. Soap.","en"),
    (23,"Whiplash","Drama|Music",2014,"Damien Chazelle",8.5,"intense|inspiring|gripping",107,"Miles Teller, J.K. Simmons","The road to greatness.","en"),
    (24,"Good Will Hunting","Drama|Romance",1997,"Gus Van Sant",8.3,"emotional|inspiring|feel-good",126,"Matt Damon, Robin Williams","Someone believes in them first.","en"),
    (25,"The Prestige","Drama|Mystery|Sci-Fi",2006,"Christopher Nolan",8.5,"mind-bending|dark|gripping",130,"Hugh Jackman, Christian Bale","Are you watching closely?","en"),
    (26,"Schindler's List","Biography|Drama|History",1993,"Steven Spielberg",9.0,"emotional|dark|gripping",195,"Liam Neeson","Whoever saves one life, saves the world.","en"),
    (27,"12 Angry Men","Crime|Drama",1957,"Sidney Lumet",9.0,"gripping|intense|inspiring",96,"Henry Fonda","Life is in their hands.","en"),
    (28,"Parasite","Drama|Thriller",2019,"Bong Joon-ho",8.5,"dark|gripping|mind-bending",132,"Song Kang-ho","Act like you own the place.","ko"),
    (29,"The Grand Budapest Hotel","Adventure|Comedy|Crime",2014,"Wes Anderson",8.1,"stylish|fun|quirky",99,"Ralph Fiennes","Check in. Have fun. Run for your life.","en"),
    (30,"Knives Out","Comedy|Crime|Drama",2019,"Rian Johnson",7.9,"fun|gripping|quirky",130,"Daniel Craig, Ana de Armas","Hell of a thing.","en"),
    (31,"The Nice Guys","Action|Comedy|Crime",2016,"Shane Black",7.4,"fun|quirky|stylish",116,"Russell Crowe, Ryan Gosling","You hired them. You're in trouble.","en"),
    (32,"What We Do in the Shadows","Comedy|Horror",2014,"Taika Waititi",7.6,"fun|quirky|dark",86,"Jemaine Clement","Some interviews are to die for.","en"),
    (33,"Superbad","Comedy",2007,"Greg Mottola",7.6,"fun|feel-good",113,"Jonah Hill, Michael Cera","It's about growing up.","en"),
    (34,"Get Out","Horror|Mystery|Thriller",2017,"Jordan Peele",7.7,"dark|intense|mind-bending",104,"Daniel Kaluuya","Just because you're invited.","en"),
    (35,"A Quiet Place","Drama|Horror|Sci-Fi",2018,"John Krasinski",7.5,"intense|atmospheric|dark",90,"Emily Blunt","If they hear you, they hunt you.","en"),
    (36,"Hereditary","Drama|Horror|Mystery",2018,"Ari Aster",7.3,"dark|intense|atmospheric",127,"Toni Collette","Every family tree hides a secret.","en"),
    (37,"The Lighthouse","Drama|Fantasy|Horror",2019,"Robert Eggers",7.5,"dark|atmospheric|mind-bending",109,"Willem Dafoe, Robert Pattinson","What secrets lie beyond the light?","en"),
    (38,"Midsommar","Drama|Horror|Mystery",2019,"Ari Aster",7.1,"dark|atmospheric|intense",148,"Florence Pugh","Let the festivities begin.","en"),
    (39,"La La Land","Drama|Music|Romance",2016,"Damien Chazelle",8.0,"emotional|feel-good|stylish",128,"Ryan Gosling, Emma Stone","Here's to the fools who dream.","en"),
    (40,"Before Sunrise","Drama|Romance",1995,"Richard Linklater",8.1,"emotional|atmospheric|feel-good",101,"Ethan Hawke, Julie Delpy","Can the greatest romance last one night?","en"),
    (41,"About Time","Drama|Fantasy|Romance",2013,"Richard Curtis",7.8,"emotional|feel-good|inspiring",123,"Domhnall Gleeson","A life lived fully.","en"),
    (42,"Her","Drama|Romance|Sci-Fi",2013,"Spike Jonze",8.0,"emotional|atmospheric|mind-bending",126,"Joaquin Phoenix","A love story for the digital age.","en"),
    (43,"Eternal Sunshine","Drama|Romance|Sci-Fi",2004,"Michel Gondry",8.3,"emotional|mind-bending|atmospheric",108,"Jim Carrey, Kate Winslet","You can erase someone from your mind.","en"),
    (44,"Your Name","Animation|Drama|Fantasy",2016,"Makoto Shinkai",8.4,"emotional|feel-good|mind-bending",106,"Ryunosuke Kamiki","What is your name?","ja"),
    (45,"Spider-Man Into the Spider-Verse","Animation|Action|Adventure",2018,"Lord/Miller",8.4,"stylish|fun|inspiring",117,"Shameik Moore","He's not the only one.","en"),
    (46,"Spirited Away","Animation|Adventure|Family",2001,"Hayao Miyazaki",8.6,"atmospheric|mind-bending|feel-good",125,"Daveigh Chase","The tunnel led Chihiro to a mysterious town.","ja"),
    (47,"Coco","Animation|Adventure|Family",2017,"Lee Unkrich",8.4,"emotional|feel-good|inspiring",105,"Anthony Gonzalez","The land of the dead never felt so alive.","en"),
    (48,"Prisoners","Crime|Drama|Mystery",2013,"Denis Villeneuve",8.1,"dark|intense|gripping",153,"Hugh Jackman, Jake Gyllenhaal","Every moment matters.","en"),
    (49,"Se7en","Crime|Drama|Mystery",1995,"David Fincher",8.6,"dark|intense|gripping",127,"Brad Pitt, Morgan Freeman","Seven deadly sins. Seven ways to die.","en"),
    (50,"The Usual Suspects","Crime|Drama|Mystery",1995,"Bryan Singer",8.5,"dark|mind-bending|gripping",106,"Kevin Spacey","The greatest trick the devil ever pulled.","en"),
    (51,"No Country for Old Men","Crime|Drama|Thriller",2007,"Coen Brothers",8.2,"dark|intense|gripping",122,"Javier Bardem, Tommy Lee Jones","There are no clean getaways.","en"),
    (52,"Lord of the Rings Fellowship","Adventure|Drama|Fantasy",2001,"Peter Jackson",8.8,"epic|atmospheric|inspiring",178,"Elijah Wood, Ian McKellen","One ring to rule them all.","en"),
    (53,"Gladiator","Action|Adventure|Drama",2000,"Ridley Scott",8.5,"epic|intense|inspiring",155,"Russell Crowe","What we do in life echoes in eternity.","en"),
    (54,"1917","Action|Drama|War",2019,"Sam Mendes",8.3,"intense|epic|gripping",119,"George MacKay","Time is the enemy.","en"),
    (55,"Dunkirk","Action|Drama|History",2017,"Christopher Nolan",7.9,"intense|atmospheric|epic",106,"Fionn Whitehead","Survival is victory.","en"),
    (56,"Mulholland Drive","Drama|Fantasy|Mystery",2001,"David Lynch",7.9,"dark|mind-bending|atmospheric",147,"Naomi Watts","A love story in the city of dreams.","en"),
    (57,"There Will Be Blood","Drama",2007,"Paul Thomas Anderson",8.2,"dark|intense|gripping",158,"Daniel Day-Lewis","When ambition meets faith.","en"),
    (58,"Moonlight","Drama|Romance",2016,"Barry Jenkins",7.4,"emotional|atmospheric|inspiring",111,"Trevante Rhodes","Who is you?","en"),
    (59,"Gone Girl","Drama|Mystery|Thriller",2014,"David Fincher",8.1,"dark|gripping|mind-bending",149,"Ben Affleck, Rosamund Pike","You don't know what you've got.","en"),
    (60,"Oldboy","Action|Drama|Mystery",2003,"Park Chan-wook",8.4,"dark|mind-bending|intense",120,"Choi Min-sik","15 years of imprisonment. 5 days of vengeance.","ko"),
    (61,"Memento","Mystery|Thriller",2000,"Christopher Nolan",8.4,"mind-bending|dark|gripping",113,"Guy Pearce","Some memories are best forgotten.","en"),
    (62,"Nightcrawler","Crime|Drama|Thriller",2014,"Dan Gilroy",7.9,"dark|intense|gripping",117,"Jake Gyllenhaal","The city shines brightest at night.","en"),
    (63,"The Social Network","Biography|Drama",2010,"David Fincher",7.8,"intense|gripping|inspiring",120,"Jesse Eisenberg","Without making a few enemies.","en"),
    (64,"Wolf of Wall Street","Biography|Comedy|Crime",2013,"Martin Scorsese",8.2,"intense|fun|gripping",180,"Leonardo DiCaprio","More. More. More.","en"),
    (65,"Inglourious Basterds","Adventure|Drama|War",2009,"Quentin Tarantino",8.3,"intense|stylish|gripping",153,"Brad Pitt, Christoph Waltz","Once upon a time in Nazi-occupied France.","en"),
    (66,"Pulp Fiction","Crime|Drama",1994,"Quentin Tarantino",8.9,"dark|stylish|gripping",154,"John Travolta, Samuel L. Jackson","You won't know the facts until you've seen the fiction.","en"),
    (67,"2001 A Space Odyssey","Adventure|Sci-Fi",1968,"Stanley Kubrick",8.3,"mind-bending|atmospheric|epic",149,"Keir Dullea","An epic drama of adventure.","en"),
    (68,"Pan's Labyrinth","Drama|Fantasy|War",2006,"Guillermo del Toro",8.2,"dark|atmospheric|mind-bending",118,"Ivana Baquero","Make-believe believes it's real.","es"),
    (69,"City of God","Crime|Drama",2002,"Fernando Meirelles",8.6,"dark|intense|gripping",130,"Alexandre Rodrigues","The beast catches you.","pt"),
    (70,"Amelie","Comedy|Romance",2001,"Jean-Pierre Jeunet",8.3,"feel-good|quirky|atmospheric",122,"Audrey Tautou","She'll change your life.","fr"),
    (71,"Burning","Drama|Mystery|Thriller",2018,"Lee Chang-dong",7.5,"atmospheric|dark|mind-bending",148,"Yoo Ah-in, Steven Yeun","The slow-burn mystery.","ko"),
    (72,"Stalker","Drama|Sci-Fi",1979,"Andrei Tarkovsky",8.1,"atmospheric|mind-bending|dark",162,"Aleksandr Kajdanovsky","Into the zone.","ru"),
    (73,"Wolfwalkers","Animation|Adventure|Family",2020,"Tomm Moore",8.1,"atmospheric|feel-good|inspiring",103,"Honor Kneafsey","Run with the wolves.","en"),
    (74,"Us","Horror|Mystery|Thriller",2019,"Jordan Peele",6.8,"dark|intense|mind-bending",116,"Lupita Nyong'o","Watch yourself.","en"),
    (75,"Game Night","Action|Comedy|Mystery",2018,"Daley/Goldstein",7.0,"fun|quirky",100,"Jason Bateman","This game is to die for.","en"),
    (76,"Zodiac","Crime|Drama|Mystery",2007,"David Fincher",7.7,"dark|gripping|atmospheric",157,"Jake Gyllenhaal","More than one way to lose your life.","en"),
    (77,"Lawrence of Arabia","Adventure|Biography|Drama",1962,"David Lean",8.3,"epic|atmospheric|inspiring",228,"Peter O'Toole","A life in a life.","en"),
    (78,"Coherence","Drama|Mystery|Sci-Fi",2013,"James Ward Byrkit",7.2,"mind-bending|atmospheric|dark",89,"Emily Baldoni","What if parallel worlds existed?","en"),
    (79,"The Witch","Drama|Horror|Mystery",2015,"Robert Eggers",6.9,"dark|atmospheric|intense",92,"Anya Taylor-Joy","Evil takes many forms.","en"),
    (80,"Drive","Crime|Drama|Thriller",2011,"Nicolas Winding Refn",7.8,"stylish|atmospheric|intense",100,"Ryan Gosling","Some heroes are real.","en"),
    (81,"Whiplash","Drama|Music",2014,"Damien Chazelle",8.5,"intense|inspiring|gripping",107,"Miles Teller","Not quite my tempo.","en"),
    (82,"Neon Genesis Evangelion","Animation|Action|Drama",1997,"Hideaki Anno",8.5,"dark|mind-bending|emotional",650,"Megumi Ogata","You are not alone.","ja"),
    (83,"The Revenant","Adventure|Drama|Thriller",2015,"Alejandro Inarritu",8.0,"intense|epic|gripping",156,"Leonardo DiCaprio","Blood lost. Life found.","en"),
    (84,"Joker","Crime|Drama|Thriller",2019,"Todd Phillips",8.4,"dark|intense|mind-bending",122,"Joaquin Phoenix","Put on a happy face.","en"),
    (85,"Oppenheimer","Biography|Drama|History",2023,"Christopher Nolan",8.3,"intense|epic|gripping",180,"Cillian Murphy","The world forever changes.","en"),
    (86,"Killers of the Flower Moon","Biography|Crime|Drama",2023,"Martin Scorsese",7.6,"dark|gripping|intense",206,"Leonardo DiCaprio","A true story of greed.","en"),
    (87,"Poor Things","Comedy|Drama|Romance",2023,"Yorgos Lanthimos",8.0,"quirky|mind-bending|stylish",141,"Emma Stone","An extraordinary adventure.","en"),
    (88,"Past Lives","Drama|Romance",2023,"Celine Song",7.8,"emotional|atmospheric|feel-good",106,"Greta Lee","What could have been.","en"),
    (89,"The Zone of Interest","Biography|Drama|History",2023,"Jonathan Glazer",7.3,"dark|atmospheric|intense",105,"Christian Friedel","Banality of evil.","de"),
    (90,"Anatomy of a Fall","Crime|Drama|Mystery",2023,"Justine Triet",7.7,"dark|gripping|mind-bending",152,"Sandra Huller","Truth is in the eye of the beholder.","fr"),
]

COLS = ["movieId","title","genres","year","director","imdb_rating",
        "mood_tags","runtime","cast","tagline","language"]
movies_df = pd.DataFrame(MOVIES, columns=COLS).drop_duplicates("movieId")

ARCHETYPES = {
    "action_fan":   ({"Action","Adventure","Thriller"},    0.60),
    "drama_lover":  ({"Drama","Biography","History","Romance"}, 0.68),
    "scifi_nerd":   ({"Sci-Fi","Fantasy","Mystery"},       0.65),
    "horror_buff":  ({"Horror","Thriller","Mystery"},      0.58),
    "comedy_fan":   ({"Comedy","Animation","Family"},      0.62),
    "cinephile":    ({"Drama","Crime","Mystery","History"},0.72),
    "world_cinema": ({"Drama","Crime","Romance","Mystery"},0.70),
    "blockbuster":  ({"Action","Adventure","Animation"},   0.55),
}

def genre_overlap(gs, preferred):
    mg = set(gs.split("|"))
    return len(mg & preferred) / max(len(mg), 1)

archs = list(ARCHETYPES.items())
n_users = 500
rows = []
for uid in range(1, n_users + 1):
    arch, (pref, bias) = archs[uid % len(archs)]
    n_rated = np.random.randint(15, 40)
    rated   = np.random.choice(movies_df["movieId"].tolist(), size=n_rated, replace=False)
    for mid in rated:
        row    = movies_df[movies_df["movieId"] == mid].iloc[0]
        ov     = genre_overlap(row["genres"], pref)
        base   = 2.0 + ov * bias * 3.0 + (row["imdb_rating"] - 7.0) * 0.45
        rating = np.clip(np.random.normal(base, 0.7), 0.5, 5.0)
        rows.append({"userId": uid, "movieId": int(mid),
                     "rating": round(rating * 2) / 2, "timestamp": uid * 1000 + int(mid)})

ratings_df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
movies_df.to_csv("data/movies.csv", index=False)
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"✅ {len(movies_df)} movies  → data/movies.csv")
print(f"✅ {len(ratings_df):,} ratings → data/ratings.csv")
print(f"   {ratings_df['userId'].nunique()} users | {len(ratings_df)/n_users:.1f} avg/user")
