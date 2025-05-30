import logging

from db import execute_query


async def apply_migrations():
    """Применяет миграции для базы данных."""
    queries = """
    create table if not exists dialogues
(
	id int auto_increment
		primary key,
	is_current tinyint(1) default 1 null,
	created_at timestamp default CURRENT_TIMESTAMP null,
	chat_id bigint not null
);

create index idx_dialogues_chat_is_current
	on dialogues (chat_id, is_current);

create table if not exists feedbacks
(
	id int auto_increment
		primary key,
	chat_id bigint not null,
	feedback text not null,
	time timestamp default CURRENT_TIMESTAMP null,
	useful tinyint(1) default 0 null
);

create index idx_feedbacks_chat_id
	on feedbacks (chat_id);

create table if not exists images
(
	id int auto_increment
		primary key,
	content mediumblob not null,
	created_at timestamp default CURRENT_TIMESTAMP null
);

create table if not exists messages
(
	id int auto_increment
		primary key,
	message text not null,
	role varchar(50) not null,
	time timestamp default CURRENT_TIMESTAMP null,
	dialogue_id int not null
);

create index idx_messages_dialogue_id
	on messages (dialogue_id);

create index idx_messages_dialogue_time
	on messages (dialogue_id, time);

create table if not exists users
(
	chat_id bigint not null
		primary key,
	invite_code varchar(36) not null,
	invited_by bigint null,
	registered_at timestamp default CURRENT_TIMESTAMP null,
	constraint invite_code
		unique (invite_code)
);

create index invited_by
	on users (invited_by);
	
alter table messages add column token_count int default 0 null;

alter table messages add column summarized tinyint(1) default 0 null;

create index idx_messages_dialogue_time_summarized
	on messages (dialogue_id, time, summarized);
	
alter table dialogues add column summary text null;

create table if not exists characters
(
	id int auto_increment
		primary key,
	name varchar(255) not null,
	card text not null,
	first_message text not null,
	user_first_message text not null,
	first_summary text not null
);

alter table dialogues add column character_id int null;
    """
    # Разбиваем запросы на отдельные команды
    for query in queries.strip().split(';'):
        query = query.strip()
        if query:
            try:
                await execute_query(query)
                logging.info(f"Миграция выполнена: {query}")
            except Exception as e:
                logging.info(f"Ошибка при выполнении миграции: {query}. Ошибка: {e}")
