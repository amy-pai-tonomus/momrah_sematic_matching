##############server_path.py##############

import time
import logging
import argsparser
from flask_restx import *
from flask import *

import visual_pollution_qa_retrieval

ns = Namespace(
	'momrah_gpt', 
	description='Visual pollution QA',
	)

args = argsparser.prepare_args()

#############

chunking_parser = ns.parser()
chunking_parser.add_argument('question', type=str, location='json')

chunking_inputs = ns.model(
	'momrah_gpt', 
		{
			'question': fields.String(example = u"What is visual pollution?")
		}
	)

@ns.route('/visual_pollution_qa')

class chunking_api(Resource):
	def __init__(self, *args, **kwargs):
		super(chunking_api, self).__init__(*args, **kwargs)
	@ns.expect(chunking_inputs)
	def post(self):		
		start = time.time()
		try:			
			args = chunking_parser.parse_args()
			#
			output = {}
			response = visual_pollution_qa_retrieval.answer_search(args['question'])
			output['question'] = args['question']
			output['answer'] = response['answer']
			output['reference'] = response['reference text']
			output['score'] = response['score']
			output['status'] = 'success'
			output['running_time'] = float(time.time()- start)
			return output, 200
		except Exception as e:
			output = {}
			output['status'] = str(e)
			output['running_time'] = float(time.time()- start)
			return output

#############


##############server_path.py##############