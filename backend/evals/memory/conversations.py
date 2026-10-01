"""Multi-turn conversations about the grounding eval's mock data.

Later turns lean on earlier ones ("that category", "the second one", "the
total you found at the start"), so an agent that lost the earlier turns has to
query again or guess. Each turn is a grounding `Question`: its expected values
come from pandas over the generated rows, not from SQL.
"""

from dataclasses import dataclass

import pandas as pd

from evals.grounding.mock_data import generate, per_video
from evals.grounding.questions import Question


@dataclass
class Conversation:
    id: str
    turns: list[Question]
    about: str = ""
    """What the follow-ups test."""


def build_conversations(df: pd.DataFrame | None = None) -> list[Conversation]:
    df = generate() if df is None else df
    videos = per_video(df)
    sizes = videos.groupby("category_name").size()
    medians = videos.groupby("category_name").views.median().sort_values(ascending=False)
    top, second = medians.index[0], medians.index[1]
    in_top = videos[videos.category_name == top].groupby("channel_title").views.sum().sort_values(ascending=False)
    channels = videos.groupby("channel_title").views.sum().sort_values(ascending=False)
    channel_category = videos.groupby("channel_title").category_name.agg(lambda s: s.mode()[0])
    channel_videos = videos.groupby("channel_title").size()
    pottery = videos[videos.category_name == "Pottery"]
    viral = pottery.sort_values("views").iloc[-1]
    days = videos.groupby("category_name").days_to_trend.median()
    more = "Speedcubing" if sizes["Speedcubing"] > sizes["Origami"] else "Origami"
    fewest = sizes.idxmin()
    biggest = sizes.idxmax()
    in_biggest = videos[videos.category_name == biggest]
    top_channel = channels.index[0]
    top20 = videos.sort_values("views", ascending=False).head(20)
    daily = df.groupby("trending_date").video_id.nunique()
    feb = daily[daily.index.str.startswith("2031-02")]
    mar = daily[daily.index.str.startswith("2031-03")]
    distinct_in = {m: df[df.trending_date.str.startswith(m)].video_id.nunique() for m in ("2031-02", "2031-03")}
    per_channel = videos.groupby("channel_title").agg(total=("views", "sum"), n=("video_id", "count"))
    smallest_channel = per_channel.n.idxmin()
    like_ratio = videos.groupby("category_name").like_ratio.mean().sort_values()
    hours = videos.groupby("publish_hour").size()
    top_videos = videos[videos.channel_title == top_channel]

    return [
        Conversation("drilldown", about="'that' refers to the previous answer; the last turn needs turn 1's ranking", turns=[
            Question("drilldown.1", "Which category has the highest median views per video, and what is that median?",
                     expect_text=[top], expect_numbers={"median": medians[top]}),
            Question("drilldown.2", "How many videos is that median based on?",
                     expect_numbers={"videos": sizes[top]}),
            Question("drilldown.3", "Which channel has the most total views in that category, counting each video once at its peak?",
                     expect_text=[in_top.index[0]], expect_numbers={"total": in_top.iloc[0]}),
            Question("drilldown.4", "Which category came second by that median, and with what value?",
                     expect_text=[second], expect_numbers={"median": medians[second]}),
        ]),
        Conversation("channels", about="'the first one', 'the second one' point into a list from turn 1", turns=[
            Question("channels.1", "Which 3 channels have the most total views, counting each video once at its peak? Give each channel's total.",
                     expect_text=list(channels.index[:3]),
                     expect_numbers={f"#{i + 1} total": channels.iloc[i] for i in range(3)}),
            Question("channels.2", "Which category are the first one's videos in?",
                     expect_text=[channel_category[channels.index[0]]]),
            Question("channels.3", "How many videos does the second one have?",
                     expect_numbers={"videos": channel_videos[channels.index[1]]}),
        ]),
        Conversation("outlier", about="follow-ups question the previous number", turns=[
            Question("outlier.1", "What are the average views per Pottery video?",
                     expect_numbers={"mean at peak": pottery.views.mean()}),
            Question("outlier.2", "That seems high. What is the median instead?",
                     expect_numbers={"median": pottery.views.median()}),
            Question("outlier.3", "Which video pulls the average up so much? Give its channel and peak views.",
                     expect_text=[viral.channel_title], expect_numbers={"peak views": viral.views}),
        ]),
        Conversation("compare", about="short follow-ups ('And for ...?', 'the two') that only make sense with the turns before", turns=[
            Question("compare.1", "What is the median number of days it takes an Origami video to start trending?",
                     expect_numbers={"median days": days["Origami"]}),
            Question("compare.2", "And for Speedcubing?",
                     expect_numbers={"median days": days["Speedcubing"]}),
            Question("compare.3", "Which of the two categories has more videos, and how many more?",
                     expect_text=[more], expect_numbers={"difference": abs(sizes["Speedcubing"] - sizes["Origami"])}),
        ]),
        Conversation("long_reference", about="the last turn points back three turns, past an unrelated one", turns=[
            Question("long_reference.1", "How many distinct videos are in the table, and how many rows does it have?",
                     expect_numbers={"videos": len(videos), "rows": len(df)}),
            Question("long_reference.2", "Which category has the fewest videos?",
                     expect_text=[fewest], expect_numbers={"videos": sizes[fewest]}),
            Question("long_reference.3", "What are the average views per Pottery video?",
                     expect_numbers={"mean at peak": pottery.views.mean()}),
            Question("long_reference.4", "Going back to the number of distinct videos you found at the start: what percentage of them are Birdwatching videos?",
                     expect_numbers={"percent": 100 * sizes["Birdwatching"] / len(videos)}),
        ]),
        Conversation("long_chat", about="a normal user: small talk, follow-ups, a topic switch, and a jump back to the start", turns=[
            Question("long_chat.1", "hey! what's in this dataset?"),
            Question("long_chat.2", "nice. how many distinct videos are there?",
                     expect_numbers={"videos": len(videos)}),
            Question("long_chat.3", "cool, which category has the most videos?",
                     expect_text=[biggest], expect_numbers={"videos": sizes[biggest]}),
            Question("long_chat.4", "and what's the average views per video there?",
                     expect_numbers={"mean at peak": in_biggest.views.mean()}),
            Question("long_chat.5", "hmm that's way higher than I expected. what's the median?",
                     expect_numbers={"median": in_biggest.views.median()}),
            Question("long_chat.6", "ok that makes sense. btw what does 'trending' actually mean on youtube?"),
            Question("long_chat.7", "anyway, different question: which channel has the most total views, counting each video once at its peak?",
                     expect_text=[top_channel], expect_numbers={"total": channels.iloc[0]}),
            Question("long_chat.8", "what category is that channel in?",
                     expect_text=[channel_category[top_channel]]),
            Question("long_chat.9", "how long does a video in that category usually take to start trending? median days is fine",
                     expect_numbers={"median days": days[channel_category[top_channel]]}),
            Question("long_chat.10", "thanks, this is really helpful!"),
            Question("long_chat.11", "oh wait, going back to the category with the most videos: how many different channels post in it?",
                     expect_numbers={"channels": in_biggest.channel_title.nunique()}),
            Question("long_chat.12", "last one: what share of all the distinct videos is that category?",
                     expect_numbers={"percent": 100 * len(in_biggest) / len(videos)}),
        ]),
        Conversation("marathon", about="24 turns: big result lists, topic switches, small talk, and questions about lists from much earlier", turns=[
            Question("marathon.1", "hi! what can you tell me about this data?"),
            Question("marathon.2", "show me the top 20 videos by peak views, with title, channel, category and views",
                     expect_text=[top20.iloc[0].channel_title], expect_numbers={"#1 views": top20.iloc[0].views}),
            Question("marathon.3", "how many of those are Beekeeping videos?",
                     expect_numbers={"beekeeping in top 20": (top20.category_name == "Beekeeping").sum()}),
            Question("marathon.4", "now list how many distinct videos were trending on each day in February 2031",
                     expect_numbers={"busiest day": feb.max()}),
            Question("marathon.5", "which day in that list was the quietest?",
                     expect_numbers={"videos": feb.min()}, expect_any_text=list(feb[feb == feb.min()].index)),
            Question("marathon.6", "cool. random question: how would you explain a median to my boss in one sentence?"),
            Question("marathon.7", "list every channel with its total peak views and its number of videos",
                     expect_text=[top_channel], expect_numbers={"#1 total": per_channel.total.max()}),
            Question("marathon.8", "which channel in that list has the fewest videos?",
                     expect_text=[smallest_channel]),
            Question("marathon.9", "what are its average peak views per video?",
                     expect_numbers={"mean": per_channel.total[smallest_channel] / per_channel.n[smallest_channel]}),
            Question("marathon.10", "thanks. let's look at categories now: what's the average like ratio (likes / views) per category, using each video at its peak?",
                     expect_text=[like_ratio.index[-1]]),
            Question("marathon.11", "which category has the lowest?",
                     expect_text=[like_ratio.index[0]]),
            Question("marathon.12", "going back to the top 20 videos from earlier: how many of them were published on a weekend?",
                     expect_numbers={"weekend": top20.publish_day.isin(["Saturday", "Sunday"]).sum()}),
            Question("marathon.13", "at which hour of the day are the most videos published, and how many videos is that?",
                     expect_numbers={"videos": hours.max()}),
            Question("marathon.14", "and what's the average like ratio for Speedcubing specifically?",
                     expect_numbers={"like ratio": like_ratio["Speedcubing"]}),
            Question("marathon.15", "ok, show me the daily distinct trending video counts for March 2031 too",
                     expect_numbers={"busiest day": mar.max()}),
            Question("marathon.16", "was March busier than February overall? compare the number of distinct videos that trended in each month",
                     expect_numbers={"february": distinct_in["2031-02"], "march": distinct_in["2031-03"]}),
            Question("marathon.17", "lol ok. do you like cats?"),
            Question("marathon.18", "back to the channel list: which channels have more than 15 videos?",
                     expect_text=list(per_channel[per_channel.n > 15].index)),
            Question("marathon.19", "for the channel with the most total views, what's the median peak views per video?",
                     expect_text=[top_channel], expect_numbers={"median": top_videos.views.median()}),
            Question("marathon.20", "how many of its videos have comments disabled?",
                     expect_numbers={"comments disabled": top_videos.comments_disabled.sum()}),
            Question("marathon.21", "summarize what we found about February in two sentences"),
            Question("marathon.22", "what was the quietest February day again?",
                     expect_any_text=list(feb[feb == feb.min()].index)),
            Question("marathon.23", "and how many videos trended on the busiest February day?",
                     expect_numbers={"videos": feb.max()}),
            Question("marathon.24", "great, thanks, that's all for today!"),
        ]),
    ]


def expectation(q: Question) -> list[str]:
    """What a turn's answer is checked for, readably: names, then label=value."""
    numbers = [f"{k}={v:,.0f}" if v >= 100 else f"{k}={v:.3g}" for k, v in q.expect_numbers.items()]
    either = [" or ".join(q.expect_any_text)] if q.expect_any_text else []
    return [*q.expect_text, *either, *numbers]


if __name__ == "__main__":
    for c in build_conversations():
        print(f"\n## {c.id}: {c.about}")
        for q in c.turns:
            print(f"  {q.id.split('.')[1]}. {q.prompt}\n     expect: {', '.join(expectation(q))}")
