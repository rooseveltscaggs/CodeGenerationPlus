import torch
import os
import tempfile
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from transformers import AutoTokenizer, AutoModelForCausalLM

code_prompt = "# this function prints hello world"
# os.system('python example_code.py')
# execfile('example_code.py')


# """ dataset keys: src, trg_prediction, reference (only trg_prediction useful) """
# def evaluate_google_mbpp(dataset, reference_path, split='test', timeout=10, return_details=False):
#     references = MBPPGoogleDataset(reference_path)
#     assert len(dataset) == len(references.raw_data[split])
tempdir = tempfile.TemporaryDirectory()
passed_information = list()

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
inputs = tokenizer(code_prompt, return_tensors="pt").to(0)
sample = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))
f = open("{tempdir.name}/code.py", "w")
f.write(str(tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])))
f.close()
#     pbar = tqdm(references.raw_data[split])
#     for i, item in enumerate(pbar):
#         if 'execution_result_full_pass' in dataset[i]:
#             passed_information.append(int(all(x[1] == True for x in dataset[i]['execution_result_full_pass'])))
#         else:
#             test_cases = item['test_list']
#             test_setups = item['test_setup_code']
#             code = dataset[i]['trg_prediction']
#             # write code to file 
#             with open(f'{tempdir.name}/code.py', 'w') as fout:
#                 print(code, file=fout)
#                 print(test_setups, file=fout)
#                 for case in test_cases:
#                     print(case, file=fout)
#                 fout.close()
    command = Command(f'python {tempdir.name}/code.py >/dev/null 2>&1')
    execution_result = (command.run(timeout=timeout) == 0)
    passed_information.append(int(execution_result))
pbar.set_description(f'{sum(passed_information)} out of {i+1} passed.')
tempdir.cleanup()
#     if return_details:
#         return passed_information
#     else:
#         return sum(passed_information) / len(passed_information)