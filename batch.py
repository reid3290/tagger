# -*- coding: utf-8 -*-
import random
import toolbox
import numpy as np
import datetime


def train(sess, placeholders, batch_size, train_step, loss,
          lr, lrv, data, dr=None, drv=None, pixels=None, pt_h=None, verbose=False):
    """
    训练单个 bucket 的模型
    :param loss:
    :param sess: tf.Session
    :param placeholders: [tf.placeholder]，总共5个，表示一个句子的字符本身、偏旁部首、2gram、3gram、对应标签
    :param batch_size:
    :param train_step: 目标 bucket 的 train_step, optimizer.apply_gradients()
    :param lr: 初始学习率
    :param lrv: 衰减后的学习率
    :param data: 当前 bucket 中的所有句子，shape=(5, bucket 中句子数量，句子长度)
    :param dr: drop_out
    :param drv: =drop_out
    :param pixels:
    :param pt_h:
    :param verbose:
    """
    # len(data)=5，表示一个句子的字符本身、偏旁部首、2gram、3gram、对应标签
    lm_targets = []
    for sentence in data[0]:
        lm_target = np.append(sentence[1:],0)
        lm_targets.append(lm_target)
    data.append(lm_targets)
    assert len(data) == len(placeholders)
    num_items = len(data)
    samples = zip(*data)
    random.shuffle(samples)
    start_idx = 0
    n_samples = len(samples)
    placeholders.append(lr)
    if dr is not None:
        placeholders.append(dr)
    if pixels is not None:
        placeholders.append(pt_h)
    while start_idx < len(samples):
        if verbose:
            print '%s : %d of %d' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), start_idx, n_samples)
        next_batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch_size = len(next_batch_samples)
        if real_batch_size < batch_size:
            next_batch_samples.extend(samples[:batch_size - real_batch_size])
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_samples])
        holders.append(lrv)
        if dr is not None:
            holders.append(drv)
        if pixels is not None:
            pt_ids = [s[0] for s in next_batch_samples]
            holders.append(toolbox.get_batch_pixels(pt_ids, pixels))
        feed_dict = {m: h for m, h in zip(placeholders, holders)}
        _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
        start_idx += batch_size


def predict(sess, model, data, dr=None, transitions=None, crf=True, decode_sess=None, scores=None, decode_holders=None,
            argmax=True, batch_size=100, pixels=None, pt_h=None, ensemble=False, verbose=False):
    en_num = None
    if ensemble:
        en_num = len(sess)
    # 输入向量是4个，字符、偏旁、2gram、3gram
    num_items = len(data)
    input_v = model[:num_items]
    if dr is not None:
        input_v.append(dr)
    if pixels is not None:
        input_v.append(pt_h)
    # 预测向量1个
    predictions = model[num_items:]
    # output = [[]]
    output = [[] for _ in range(len(predictions))]
    samples = zip(*data)
    start_idx = 0
    n_samples = len(samples)
    if crf:
        trans = []
        for i in range(len(predictions)):
            if ensemble:
                en_trans = 0
                for en_sess in sess:
                    en_trans += en_sess.run(transitions[i])
                trans.append(en_trans/en_num)
            else:
                trans.append(sess.run(transitions[i]))
    while start_idx < n_samples:
        if verbose:
            print '%d' % (start_idx*100/n_samples) + '%'
        next_batch_input = samples[start_idx:start_idx + batch_size]
        batch_size = len(next_batch_input)
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_input])
        if dr is not None:
            holders.append(0.0)
        if pixels is not None:
            pt_ids = [s[0] for s in next_batch_input]
            holders.append(toolbox.get_batch_pixels(pt_ids, pixels))
        # length_holder = tf.cast(tf.pack(holders[0]), dtype=tf.int32)
        # length = tf.reduce_sum(tf.sign(length_holder), reduction_indices=1)
        length = np.sum(np.sign(holders[0]), axis=1)
        length = length.astype(int)
        if crf:
            assert transitions is not None and len(transitions) == len(predictions) and len(scores) == len(decode_holders)
            for i in range(len(predictions)):
                if ensemble:
                    en_obs = 0
                    for en_sess in sess:
                        en_obs += en_sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                    ob = en_obs/en_num
                else:
                    ob = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                # trans = sess.run(transitions[i])
                pre_values = [ob, trans[i], length, batch_size]
                assert len(pre_values) == len(decode_holders[i])
                max_scores, max_scores_pre = decode_sess.run(scores[i], feed_dict={i: h for i, h in zip(decode_holders[i], pre_values)})
                output[i].extend(toolbox.viterbi(max_scores, max_scores_pre, length, batch_size))
        elif argmax:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = np.argmax(pre, axis=2)
                pre = pre.tolist()
                pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        else:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = pre.tolist()
                pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        start_idx += batch_size
    return output


