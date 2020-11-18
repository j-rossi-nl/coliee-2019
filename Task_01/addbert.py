from basics import ColieeVectorizer, ColieeData
d = ColieeData('text_summarized_200.csv')

v1 = ColieeVectorizer(data=d)
v1.parameters['bert_server'] = 'ilps-gpu6'
v1.prepare_bert()
v1.to_file('bert_from_summarized_200.save')
