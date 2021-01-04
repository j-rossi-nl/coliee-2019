import os
import glob

from scrapy import Spider, Request
from coliee.items import AggItem, TextItem, Task_04_Item


class AggSpider(Spider):
	"""
	A spider to only get aggregated statistics
	"""
	name = 'agg_coliee_spider'
	start_urls = ['file:///home/juju/PycharmProjects/COLIEE_2019/data/Task_01/train_data/task1.xml']

	def parse(self, response):
		task_2_parse = {
			'1': self.parse_task_1,
			'2': self.parse_task_2,
		}

		# Check which task it is, Call the right parser
		task = response.xpath('/COLIEE/attribute::task').get()
		yield Request(response.url, callback=task_2_parse[task], dont_filter=True)



	def parse_task_1(self, response):
		# Loop through all cases
		cases = response.xpath('.//instance')
		for case in cases:
			# Get the case itself
			case_id = case.xpath('./attribute::id').get()
			case_file = case.xpath('./query/text()').get()

			# Get the list of relevant cases
			relevant_cases = case.xpath('./cases_noticed/text()').get().split(sep=',')

			# Loop through the list of candidate cases
			candidate_cases = case.xpath('.//candidate_case')
			for candidate_case in candidate_cases:
				candidate_id = candidate_case.xpath('./attribute::id').get()
				candidate_file = candidate_case.xpath('./text()').get()

			item = AggItem()
			item['id'] = case_id
			item['num_rel'] = len(relevant_cases)
			item['num_can'] = len(candidate_cases)
			yield item

	def parse_task_2(self, response):
		pass



class TextSpider(Spider):
	"""
	A spider to get all the text in one csv file
	"""
	name = 'text_coliee_spider'
	root_folder = '/home/juju/PycharmProjects/COLIEE_2019/data/Task_01/test_data'
	start_urls = ['file://{}'.format(os.path.join(root_folder, 'task1_test_golden-labels.xml'))]

	def parse(self, response):
		task_2_parse = {
			'1': self.parse_task_1,
			'2': self.parse_task_2,
		}

		# Check which task it is, Call the right parser
		task = response.xpath('/COLIEE/attribute::task').get()
		yield Request(response.url, callback=task_2_parse[task], dont_filter=True)



	def parse_task_1(self, response):
		# Loop through all cases
		cases = response.xpath('.//instance')
		for case in cases:
			# Get the case itself
			case_id = case.xpath('./attribute::id').get()
			case_file = case.xpath('./query/text()').get()
			with open(os.path.join(self.root_folder, case_file), 'r') as case_content:
				case_text = case_content.read()

			# Get the list of relevant cases
			# If it's a test set, this will be an empty list
			relevant_cases_txt = case.xpath('./cases_noticed/text()')
			relevant_cases = relevant_cases_txt.get().split(sep=',') if len(relevant_cases_txt) > 0 else []

			# Loop through the list of candidate cases
			candidate_cases = case.xpath('.//candidate_case')
			for candidate_case in candidate_cases:
				candidate_id = candidate_case.xpath('./attribute::id').get()
				candidate_file = candidate_case.xpath('./text()').get()
				with open(os.path.join(self.root_folder, case_id, 'candidates', candidate_file), 'r') as candidate_content:
					candidate_text = candidate_content.read()
				candidate_is_noticed = candidate_id in relevant_cases

				item = TextItem()
				item['case_id'] = case_id
				item['case_text'] = case_text
				item['candidate_id'] = candidate_id
				item['candidate_text'] = candidate_text
				item['candidate_is_noticed'] = candidate_is_noticed
				yield item

	def parse_task_2(self, response):
		pass


class Task_04_Spider(Spider):
	"""
	Preparing the dataset for training purpose
	"""
	name = 'task_04_coliee_spider'
	root_folder = '/home/juju/PycharmProjects/COLIEE_2019/data/Task_03/train_data'
	start_urls = ['file://{}'.format(file_path) for file_path in glob.glob(os.path.join(root_folder, 'riteval_*.xml'))]

	labels = {'Y': 1, 'N': 0}

	def parse(self, response):
		pairs = response.xpath('.//pair')
		for pair in pairs:
			# Get the pair itself
			query_id = pair.xpath('./attribute::id').get()
			law_text = pair.xpath('./t1/text()').get().replace('\n', ' ')
			statement = pair.xpath('./t2/text()').get().replace('\n', ' ')
			label =	Task_04_Spider.labels[pair.xpath('./attribute::label').get()]

			item = Task_04_Item()
			item['query_id'] = query_id
			item['law_text'] = law_text
			item['statement'] = statement
			item['label'] = label
			yield item
