from cs336_basics.tokenizer import Tokenizer, parse_tokenizer_arguments
import multiprocessing


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_tokenizer_arguments()
    if args.input_merge_path:
        tokenizer = Tokenizer.from_files(
            args.input_vocab_path,
            args.input_merge_path,
            special_tokens=args.special_tokens,
        )
    else:
        tokenizer = Tokenizer(special_tokens=args.special_tokens)
    if args.train_path and args.vocab_size:
        tokenizer.fit(
            vocab_size=args.vocab_size,
            file_name=args.train_path,
            verbose=True,
            num_processes=5,
        )
        if args.output_merge_path:
            tokenizer.to_file(args.output_vocab_path, args.output_merge_path)
    if args.data_path and args.data_ouptut_path:
        if not args.decode:
            tokenizer.encode_to_stream(
                args.data_path,
                args.data_ouptut_path,
                read_size=4096 * 10,
                num_processes=5,
            )
        else:
            tokenizer.decode_to_stream(
                args.data_path, args.data_ouptut_path, read_size=4096 * 10
            )


if __name__ == "__main__":
    main()
