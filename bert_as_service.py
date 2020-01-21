from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


def get_model(TUNED_FLAG=False):
    args = [
            '-model_dir', 'english_L-12_H-768_A-12/',
            '-port', '5555',
            '-port_out', '5556',
            '-max_seq_len', 'NONE',
            '-mask_cls_sep',
            'num_worker', '4',
            '-cpu',
            ]
    if TUNED_FLAG == True:
        args.extend([
            '-tuned_model_dir', '/tmp/mrpc_output/',
            '-ckpt_name', 'model.ckpt-343',
            ])

    bert_args = get_args_parser().parse_args(args)
    server = BertServer(bert_args)
    server.start()
    BertServer.shutdown(port=5555)


class BertEmbeddings(object):

    def __init__(self, ip='localhost', port=5555, port_out=5556):
        self.server = BertClient(ip=ip, port=port, port_out=port_out)

    def get_embeddings(self, text):
        emb = self.server.encode(text)
        return emb
