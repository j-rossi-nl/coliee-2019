# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class AggItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    id = scrapy.Field()
    num_rel = scrapy.Field()
    num_can = scrapy.Field()


class TextItem(scrapy.Item):
    case_id = scrapy.Field()
    case_text = scrapy.Field()
    candidate_id = scrapy.Field()
    candidate_text = scrapy.Field()
    candidate_is_noticed = scrapy.Field()

    def __repr__(self):
        return ''


class Task_04_Item(scrapy.Item):
    query_id = scrapy.Field()
    law_text = scrapy.Field()
    statement = scrapy.Field()
    label = scrapy.Field()
