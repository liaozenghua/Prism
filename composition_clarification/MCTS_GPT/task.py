import random
from tasks.science import SearchTask
from MCTS_GPT.base import treeNode
from models_GPT.get_response import *
from MCTS_GPT.mcts import MCTS
from utils_file.verify_MATH import exact_match_score, grade_answer, extract_answer
from utils_file.verify_llm import llm_verify
from utils_file.solution_summary_extractor import extract_summary_from_solution
from models_GPT.model import *
from MCTS_GPT.prompts import *

latest_grads = [None]
latest_embeddings = [None]
def forward_hook(module, input, output):
    latest_embeddings[0] = output.detach()
    output.register_hook(lambda grad: latest_grads.__setitem__(0, grad))


def get_word_indices(text_ids, tokenizer):
    all_word_indices = []
    word_indices = [0]
    lookback = 1
    for t in range(1, len(text_ids)):
        if len(tokenizer.decode(text_ids[t]).strip()) == 0:
            word_indices.append(t)
            lookback += 1 # increase lookback in case token is empty
        elif len(tokenizer.decode(text_ids[t-lookback:t+1]).split()) == 1:
            word_indices.append(t)
            lookback = 1
        else:
            all_word_indices.append(torch.tensor(word_indices))
            word_indices = [t]
            lookback = 1
    all_word_indices.append(torch.tensor(word_indices))

    return all_word_indices



class MCTS_Task(SearchTask):
    def __init__(self, data, rag,model_type, model,value_method='gpt', branch=3, end_gate=0.7, roll_policy='greedy',
                 roll_branch=1, roll_forward_steps=3, time_limit=None, iteration_limit=50, exploration_constant=0.7,
                 alpha=0.5, inf=1.0, temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, use_reflection='simple', low=0, high=1,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='zh', weighted_verify=False):
        super().__init__(data, model_type, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.model_type=model_type
        self.model=model
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify
        self.rag=rag

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def get_response(self,y,proposal,step_n):
        if self.propose_method == 'mistral' or self.propose_method == 'llama':
            prompt = self.get_response_promt_wrap_mistral(self.question, y,proposal,step_n)
            response = get_proposal(prompt, self.model_type, self.temperature, self.max_tokens, self.seed,
                                        self.max_length,
                                        self.truncation, self.do_sample, self.max_new_tokens)
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.replace('model:','').replace('Model:','').replace('user:','').replace('User:','').strip()
        p = p.strip()

        if p=='':
            print('用户回答输出格式有误！\n')
            return self.get_response(y,proposal,step_n)
        else:
            print(f'标准化后新的用户回答:{p}\n')
            return 'User: '+ p


    def get_answer(self,y,qa,step_n):
        if self.model_type == 'mistral' or self.propose_method == 'llama':
            prompt = self.get_answer_promt_wrap_mistral(self.question, y,qa,step_n)
            answer = local_answer_model(prompt, max_length=self.max_length, truncation=self.truncation,
                                             do_sample=False,
                                             max_new_tokens=1024, temperature=1.0)


        answer_logits=get_logits(prompt, answer,inference_model, inference_tokenizer)
        answer_logits['text']=answer
        return answer_logits

    def get_next_step(self, y, step_n):
        user_prompt = clarify_user.format(self.question,y,', '.join(self.rag['hierarchical_elements'][step_n-1]))

        if self.model_type=='deepseek':
            # Get deepseek response
            response = deepseek(
                clarify_system,
                user_prompt,
                self.model,
                temperature=1.3
            ).choices[0].message.content
        elif self.model_type=='gpt':
            response = gpt(
                clarify_system,
                user_prompt,
                self.model,
                temperature=1.3
            ).choices[0].message.content


        # p = p.replace('model:','').replace('Model:','').replace('user:','').replace('User:','').strip()


        if response=='':
            print('意图理解问题输出格式有误！\n')
            return self.get_next_step(y, step_n)
        else:
            print(f'标准化后新的意图问题:{response}\n')
            return 'Model: '  + response

    def get_next_step_use_reflection(self, y, step_n, reflection):  # 暂不支持 case-prompt
        if self.model_type == 'gpt' or self.model_type == 'local':
            propose_prompt = self.zero_single_propose_wrap_use_reflection_gpt(self.question, y, step_n, reflection,
                                                                              self.lang)
        else:
            propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection,
                                                                          self.lang)
        response = get_proposal(propose_prompt, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        if len(response) > 5:
            response = response[:5]

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = '步骤' + str(step_n) + ':' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    print('输出步骤过短！\n')
                    return ''
                if p_[1:] in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = '步骤' + str(step_n) + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                print('输出格式有误！\n')
                return ''

        else:
            if "Next step:" in p:
                stp = p.split('Next step:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif "Step" in p and ":" in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('Step')[0].strip()
                if len(p_) < 4:
                    print('输出步骤过短！\n')
                    return ''
                p_ = p_[1:].strip()
                if p_ in y:
                    print('输出步骤重复！\n')
                    return ''

                revised_ = 'Step ' + str(step_n) + ': ' + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            else:
                print('输出格式有误！\n')
                return ''

    def get_simple_reflection(self, y, step_n):
        if step_n >=len(self.rag['hierarchical_elements']):
            return '<end>'


        prompt_user=reflection_user.format(self.question) + '\n' + y + '\nOutput:' if y != None else reflection_user.format(self.question) + '\nOutput:'
        response = []
        while not response:
            if self.model_type == 'deepseek':
                # Get deepseek response
                response = deepseek(
                    reflection_system,
                    prompt_user,
                    'deepseek-chat',
                    temperature=0.0
                )
            elif self.model_type == 'gpt':
                response = gpt(
                    reflection_system,
                    prompt_user,
                    self.model,
                    temperature=0.0
                )
        reflection=response.choices[0].message.content

        if 'unsolved' in reflection:
            print('标准化后的意见: <continue>\n')
            return '<continue>'
        elif 'solved' in reflection:
            print('标准化后的意见: <end>\n')
            return '<end>'
        else:
            print('标准化后的意见: <continue>\n')
            return '<continue>'

    def get_reflection(self, y, step_n):
        if self.model_type in ['local', 'mistral', 'llama'] and self.lang == 'en':
            if 'answer is' in y or '\\boxed' in y:
                return '<end>'

        if self.lang == 'zh':
            if self.model_type == 'gpt' or self.propose_method == 'local':
                reflection_prompt = self.single_reflection_wrap_gpt(self.question, y, step_n)
            elif self.model_type == 'llama':
                reflection_prompt = self.single_reflection_wrap_llama(self.question, y, step_n)
            else:
                reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)
        else:
            reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self.model_type, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, self.max_new_tokens)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return ''

        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        if self.lang == 'zh':
            if '已解决' in p or '已经解决' in p:
                if step_n > 1:
                    print('此步问题已解决，停止下探。\n')
                    return '<end>'
                else:
                    return ''

            if '意见:' not in p:
                print('输出格式有误！\n')
                return ''
            revised_ = p.split('意见:')[1]
            print(f'标准化后的意见:{revised_}\n')
            return revised_

        else:
            if 'Problem solved' in p:
                print('标准化后的意见: <end>\n')
                return '<end>'
            else:
                if 'Analysis:' not in p:
                    print('输出格式有误！\n')
                    return ''
                revised_ = p.split('Analysis:')[1].strip()
                print(f'标准化后的意见:{revised_}\n')
                return revised_



    def get_step_value(self, y, answer):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        initial_generation_text=answer['text']
        initial_generation_ids=answer['output_id']
        generation_logits =answer['logits'].to(dtype=torch.float32)

        probs = generation_logits.softmax(-1)
        gen_probs = torch.gather(probs, 1, torch.tensor(initial_generation_ids, dtype=torch.long).unsqueeze(1)).squeeze(1)
        llm_word_indices = get_word_indices(text_ids=initial_generation_ids, tokenizer=inference_tokenizer)
        word_probs =torch.vstack([torch.exp(torch.log(gen_probs[word_token_indices]).mean()) for word_token_indices in llm_word_indices])


        additional_generated_text = []
        num_added_gens = 0
        handle = deberta_model.deberta.embeddings.word_embeddings.register_forward_hook(forward_hook)
        ce_loss_fn = torch.nn.CrossEntropyLoss()

        input_text='User: '+self.question+y+'\nResponse:'

        encoded_question = deberta_tokenizer.encode(input_text, padding=True, return_tensors='pt').squeeze()[:-1]  # remove SEP token (last)，得到问题的token id
        encoded_answer = deberta_tokenizer.encode(' ' + initial_generation_text, padding=True,
                                                  return_tensors='pt').squeeze()[1:-1]

        all_word_indices = get_word_indices(text_ids=encoded_answer, tokenizer=deberta_tokenizer)

        qa_initial = input_text + initial_generation_text
        input_sequence = qa_initial + ' [SEP] ' + qa_initial  ##NLI模型的输入，两个相同的问题答案组合
        model_input = [input_sequence]
        encoded_input = deberta_tokenizer(model_input, return_tensors='pt', padding=True).to('cuda')  ##NLI模型编码
        deberta_model.zero_grad()
        prediction = deberta_model(**encoded_input)['logits']

        target = torch.tensor([0] + [0] * len(additional_generated_text)).to('cuda')  ##目标是让模型认为他是矛盾的时候的梯度，所以标签要设置为0
        loss = ce_loss_fn(prediction, target)  ##计算交叉熵
        loss.backward()  ##反向传播


        assert encoded_input["input_ids"].shape[1] == latest_grads[0].shape[1] == latest_embeddings[0].shape[1]

        if encoded_question.tolist() != encoded_input["input_ids"][ 0, :len(encoded_question)].tolist():
            print(
                f"Error: {encoded_question.tolist()} vs. {encoded_input['input_ids'][0, :len(encoded_question)].tolist()}")
            return False
        for word, word_token_indices in zip(initial_generation_text.split(), all_word_indices):  ### 检查单词和token是否匹配
            if word.strip() != deberta_tokenizer.decode(encoded_answer[word_token_indices]).strip():
                print(
                    f'Error: words do not match ({word.strip()} vs. {deberta_tokenizer.decode(encoded_answer[all_word_indices]).strip()})')
                return False


        handle.remove()  ## 移除钩子
        deberta_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.no_grad():

            consider_gradients_from_both_sides = False  # set accordingly (similar empirical performance, thus set to False for more efficiency)# 是否考虑双向梯度（设置为False以提高效率）

            qa1_gradients = latest_grads[0][:, len(encoded_question):len(encoded_question) + len(encoded_answer),:]  ##生成文本的梯度
            qa1_embeddings = latest_embeddings[0][:, len(encoded_question):len(encoded_question) + len(encoded_answer),:]  ##生成文本的embedding
            qa1_attributions = qa1_gradients * qa1_embeddings  ##归因分数计算
            assert qa1_gradients.shape == qa1_embeddings.shape

            token_attributions = torch.abs(qa1_attributions)
            # (1) calculate attribution scores
            all_word_gradient_magnitudes = []
            word_attributions = torch.vstack([token_attributions[0, word_token_indices, :].mean(dim=0) for word_token_indices in all_word_indices])  ##合并同一个单词的不同token
            assert word_attributions.shape[0] == len(initial_generation_text.split())

            all_word_gradient_magnitudes.append(torch.norm(word_attributions, dim=-1).tolist())  ##计算每个单词的归因得分
            # word_gradient_magnitude.shape = [num_words]

            word_gradient_magnitude = torch.tensor(all_word_gradient_magnitudes).mean(dim=0)  ##归因得分的tensor版
            assert word_gradient_magnitude.shape[0] == len(all_word_indices)
            intent_score=torch.div(torch.sub(word_gradient_magnitude, word_gradient_magnitude.min()),word_gradient_magnitude.max() - word_gradient_magnitude.min())
            threshold = torch.mean(intent_score)+torch.var(intent_score)
            bool_tensor=intent_score>threshold


            value_list=[i*intent_score[id]*bool_tensor[id] for id,i in enumerate(word_probs)]

            valid_value=torch.tensor(value_list)[bool_tensor]**0.5
            value=torch.exp(torch.log(valid_value).mean())
        return value




        # if self.value_method == 'local':
        #         if self.lang == 'zh':
        #             prompt_answer = '问题:' + self.question + '\n步骤:\n' + '【答案】' + y
        #         else:
        #             prompt_answer = 'Problem: ' + self.question + '\nSolution:\n' + y
        #         value = get_value(prompt_answer, self.value_method, self.temperature, self.max_tokens, self.seed,
        #                           self.max_length, self.low, self.high)
        #         print(f'获得评分:{value}\n')
        #         self.value_cache.update({y: value})
        #         return value
        #
        # else:
        #     prompt = self.value_prompt_wrap(self.question, y)
        #     response = get_value(prompt, self.value_method, self.temperature, self.max_tokens, self.seed,
        #                          self.max_length, self.low, self.high)
        #     value = self.value_outputs_unwrap(response, self.low, self.high)
        #     print(f'获得评分:{value}\n')
        #     self.value_cache.update({y: value})
        #     return value

    def get_summary(self, y):
        if self.lang == 'zh':
            if self.evaluate == 'scibench':
                prompt = self.evaluate_summary_prompt_wrap(self.question, y)
            elif self.evaluate == 'scieval':
                prompt = self.general_evaluate_summary_prompt_wrap(self.question, y)
            else:
                prompt = self.summary_prompt_wrap(self.question, y)

            response = get_proposal(prompt, self.model_type, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)

            if not response:
                print('获得综述失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            p = p.strip()

            if self.evaluate:
                if len(p) < 1:
                    print('获得综述过短！\n')
                    return ''

                if '综上所述，最终答案是:' not in p:
                    summ = '综上所述，最终答案是:' + p
                    print(f'获得综述:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

            else:
                if len(p) < 1:
                    print('获得综述过短！\n')
                    return ''

                p = p.replace('综上所述,', '综上所述，')
                if '综上所述，' not in p:
                    summ = '综上所述，' + p
                    print(f'获得综述:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，' + p.split('综上所述，')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

        else:
            prompt = self.MATH_summary_prompt_wrap(self.question, y)
            response = get_proposal(prompt, self.model_type, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            if not response:
                print('获得综述失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _
            summ = p.strip()
            print(f'获得综述:{summ}\n')

            return summ

    def get_MATH_summary(self, y):
        prompt = self.MATH_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self.model_type, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('获得综述失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'获得综述:{p}\n')
        return p

    def verify_end_nodes(self, root):
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()
        flag = False
        for leaf in end_leaf_nodes:
            leaf.on_final_route = True
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(leaf.y)
                else:
                    summ = self.get_summary(leaf.y)
                if summ:
                    leaf.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(leaf.y)
                leaf.summary = summ

            if self.verify_method == 'string':
                result = exact_match_score(summ, self.answer)
            else:
                result = llm_verify(summ, self.answer)
            if result:
                if self.reward_model_type == 'vm':
                    leaf.min_steps_to_correct = 1
                else:
                    leaf.he = 1
                flag = True
        return flag, end_leaf_nodes

    def get_final_solution(self, root, weighted):  # for evaluation
        if self.reward_model_type == 'vm':
            end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
        else:
            end_leaf_nodes = root.get_all_end_root_nodes_prm()

        if not end_leaf_nodes or not weighted:
            if not end_leaf_nodes:
                best_node, best_V = root.getBestV()
            else:
                sorted_nodes = sorted(end_leaf_nodes, key=lambda x: x.V, reverse=True)
                best_node = sorted_nodes[0]
            solution = best_node.y
            cnt = 5
            summ = ''
            while cnt:
                if self.verify_method == 'string':
                    summ = self.get_MATH_summary(solution)
                else:
                    summ = self.get_summary(solution)
                if summ:
                    best_node.summary = summ
                    break
                else:
                    cnt -= 1
            if not summ:
                summ = extract_summary_from_solution(solution)
                best_node.summary = summ
            return solution, summ

        else:
            all_answers = {}  # {answer: [solution, summ, value]}
            for leaf in end_leaf_nodes:
                cnt = 5
                summ = ''
                while cnt:
                    if self.verify_method == 'string':
                        summ = self.get_MATH_summary(leaf.y)
                    else:
                        summ = self.get_summary(leaf.y)
                    if summ:
                        leaf.summary = summ
                        break
                    else:
                        cnt -= 1
                if not summ:
                    summ = extract_summary_from_solution(leaf.y)
                    leaf.summary = summ

                extracted_answer = extract_answer(summ)
                if extracted_answer in all_answers.keys():
                    all_answers[extracted_answer][2] += leaf.V
                else:
                    all_answers[extracted_answer] = [leaf.y, summ, leaf.V]

            best_answer = max(all_answers.values(), key=lambda x: x[2])
            solution = best_answer[0]
            summ = best_answer[1]
            return solution, summ

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':
            if self.sample_value != 'full':
                if self.evaluate == 'scibench':  # SciBench style
                    solution = node.y
                    summary = self.get_summary(solution)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summary,
                                    'finish': finish}
                    if self.sample_value == 'simple':
                        node.trace_route()
                        new_value_samples = node.get_new_value_samples()
                        final_answer.update({'value_samples': new_value_samples})
                else:  # MATH style
                    solution = node.y
                    # cnt = 5
                    answer = node.answer['text']
                    # while cnt:
                    #     if self.verify_method == 'string':
                    #         summ = self.get_MATH_summary(solution)
                    #     else:
                    #         summ = self.get_summary(solution)
                    #     if summ:
                    #         node.summary = summ
                    #         break
                    #     else:
                    #         cnt -= 1

                    # if not summ:
                    #     summ = extract_summary_from_solution(solution)
                    #     node.summary = summ

                    # result = exact_match_score(summ, self.answer)
                    final_answer = {'instruction': self.question, 'qa': solution, 'answer': answer, 'finish': finish}
                return final_answer, root
            else:
                if not self.evaluate:  # generate only
                    assert self.answer is not None, 'Answer is None!\n'
                    flag, end_leaf_nodes = self.verify_end_nodes(root)

                    # extract policy data
                    new_policy_samples = []
                    for leaf in end_leaf_nodes:
                        solution = leaf.y
                        summ = leaf.summary
                        correct = True if leaf.min_steps_to_correct == 1 else False
                        new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                        new_policy_samples.append(new_policy_sample)

                    # extract value data
                    if flag:
                        new_value_samples = root.get_full_value_samples_vm(end_leaf_nodes)
                    else:
                        new_value_samples = []
                    final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                                    'value_samples': new_value_samples, 'real_answer': self.answer}
                    return final_answer, root
                else:
                    assert self.answer is not None, 'Answer is None!\n'
                    solution, summ = self.get_final_solution(root, self.weighted_verify)
                    if not summ:
                        result = False
                    else:
                        result = exact_match_score(summ, self.answer)
                    final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                    'accurate': result, 'real_answer': self.answer}
                    return final_answer, root

        # prm (only sample generation available now)
        else:
            assert self.sample_value, 'Only sampling is supported for prm!\n'
            assert self.answer is not None, 'Answer is None!\n'
            flag, end_leaf_nodes = self.verify_end_nodes(root)

            # extract policy data
            new_policy_samples = []
            for leaf in end_leaf_nodes:
                solution = leaf.y
                summ = leaf.summary
                correct = True if leaf.he == 1 else False
                new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
                new_policy_samples.append(new_policy_sample)

            # extract value data
            if flag:
                new_value_samples = root.get_full_value_samples_prm(end_leaf_nodes)
            else:
                new_value_samples = []
            final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
                            'value_samples': new_value_samples, 'real_answer': self.answer}
            return final_answer, root
