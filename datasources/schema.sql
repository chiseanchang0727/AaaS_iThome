-- Table the agent queries. Columns mirror the cleaned CSV.
--
-- No natural primary key: 50 rows share (video_id, trending_date), so the load
-- is a full reload (TRUNCATE + COPY) rather than an upsert.

CREATE TABLE IF NOT EXISTS videos (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id           text NOT NULL,
    title              text,
    channel_title      text,
    category_id        int,
    category_name      text,
    country            text,
    publish_time       timestamptz,
    trending_date      date,
    publish_hour       smallint,
    publish_day        text,
    days_to_trend      double precision,
    views              bigint,
    likes              bigint,
    dislikes           bigint,
    comment_count      bigint,
    like_ratio         double precision,
    dislike_ratio      double precision,
    like_dislike_ratio double precision,
    title_length       int,
    title_caps_ratio   double precision,
    tag_count          int,
    comments_disabled  boolean,
    ratings_disabled   boolean
);

CREATE INDEX IF NOT EXISTS videos_channel_title_idx ON videos (channel_title);
CREATE INDEX IF NOT EXISTS videos_trending_date_idx ON videos (trending_date);
