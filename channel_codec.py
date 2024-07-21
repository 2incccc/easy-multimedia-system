import numpy as np
import argparse

def conv_encode(data, G):
    """
    卷积编码器
    :param data: 输入二进制数据
    :param G: 生成多项式
    :return: 编码后的二进制数据
    """
    k, n = G.shape
    m = np.max([len(np.binary_repr(g)) for g in G[0]])
    state = np.zeros(m, dtype=int)
    encoded_data = []
    for bit in data:
        state = np.roll(state, 1)
        state[0] = bit
        encoded_bits = []
        for g in G[0]:
            g_bits = np.array([int(x) for x in np.binary_repr(g).zfill(m)], dtype=int)
            encoded_bits.append(np.sum(state * g_bits) % 2)
        encoded_data.extend(encoded_bits)
    return np.array(encoded_data, dtype=int)

def viterbi_decode(received_data, G, data_length):
    """
    维特比解码器
    :param received_data: 接收到的编码数据
    :param G: 生成多项式
    :param data_length: 原始数据长度
    :return: 解码后的二进制数据
    """
    k, n = G.shape
    m = np.max([len(np.binary_repr(g)) for g in G[0]])
    state_count = 2**(m-1)
    path_metrics = np.zeros(state_count, dtype=float)
    paths = {i: [] for i in range(state_count)}
    trellis = []

    for i in range(0, len(received_data), n):
        received_bits = received_data[i:i+n]
        branch_metrics = np.zeros((state_count, state_count), dtype=float)
        for current_state in range(state_count):
            for input_bit in range(2):
                next_state = (current_state >> 1) | (input_bit << (m-2))
                g_bits = np.array([int(x) for x in np.binary_repr(G[0][input_bit], width=n)], dtype=int)
                branch_metrics[current_state, next_state] = np.sum(np.abs(received_bits - g_bits))

        trellis.append(branch_metrics)
        new_path_metrics = np.full(state_count, np.inf)
        new_paths = {i: [] for i in range(state_count)}

        for current_state in range(state_count):
            for next_state in range(state_count):
                metric = path_metrics[current_state] + branch_metrics[current_state, next_state]
                if metric < new_path_metrics[next_state]:
                    new_path_metrics[next_state] = metric
                    new_paths[next_state] = paths[current_state] + [next_state & 1]

        path_metrics = new_path_metrics
        paths = new_paths

    best_path = paths[np.argmin(path_metrics)]
    return np.array(best_path[:data_length], dtype=int)

def awgn(input_signal, SNR_dB):
    """
    AWGN信道
    :param input_signal: 输入信号
    :param SNR_dB: 信噪比
    :return: 加噪后的信号
    """
    signal_power = np.mean(input_signal ** 2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(input_signal))
    return input_signal + noise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the input binary file")
    parser.add_argument("output", help="Path to the output binary file")
    parser.add_argument("--snr", type=float, default=5, help="SNR in dB")
    args = parser.parse_args()

    G = np.array([[0b111, 0b101]])
    data_length = 100  # Example data length, should match actual data length

    with open(args.input, 'rb') as f:
        compressed_data = np.frombuffer(f.read(), dtype=np.uint8)

    bit_stream = np.unpackbits(compressed_data)
    encoded_data = conv_encode(bit_stream, G)
    received_signal = awgn(encoded_data, args.snr)
    decoded_data = viterbi_decode(received_signal, G, len(bit_stream))
    decoded_bytes = np.packbits(decoded_data[:len(bit_stream)])

    with open(args.output, 'wb') as f:
        f.write(decoded_bytes)

if __name__ == "__main__":
    main()
