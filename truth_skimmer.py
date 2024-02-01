from truth_dumper import dump_truth_info
import click

@click.command()
@click.argument('input_data', type=click.Path(exists=True), nargs=-1)
def main(input_data):
    for i in input_data:
        dump_truth_info(i)

if __name__ == "__main__":
    main()