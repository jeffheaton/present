import re

rx_infobox = re.compile(r'^{{Infobox\s([A-Za-z0-9_]+)\s*\|')

TEXT = """
{{Short description|45th president of the United States from 2017 to 2021}}
{{other uses}}
{{pp-move-indef}}
{{pp-dispute|small=yes}}
<!-- DO NOT CHANGE this hatnote without prior consensus; see [[Talk:Donald Trump#Current consensus]], item 17. -->
{{Use mdy dates|date=March 2022}}
{{Use American English|date=November 2020}}
{{Infobox officeholder
|image=Donald Trump official portrait.jpg<!-- DO NOT CHANGE the picture without prior consensus; see [[Talk:Donald Trump#Current consensus]], item 1. -->
|alt=Official White House presidential portrait. Head shot of Trump smiling in front of the U.S. flag, wearing a dark blue suit jacket with American flag lapel pin, white shirt, and light blue necktie.
|caption=Official portrait, 2017
|order=45th<!-- DO NOT ADD A LINK. Please discuss any proposal on the talk page first. Most recent discussion at [[Talk:Donald Trump/Archive 65#Link-ifying "45th" in the Infobox?]] had a weak consensus to keep the status-quo of no link. -->
|office=President of the United States
|vicepresident=[[Mike Pence]]
"""

lst = rx_infobox.findall(TEXT)
print(lst)