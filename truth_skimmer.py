from truth_dumper import dump_truth_info
import click

@click.command()
@click.argument('input_data', type=click.Path(exists=True), nargs=-1)
@click.option('-e', '--energy-cut', type=float, default=0)
def main(input_data, energy_cut):
    for i in input_data:
        dump_truth_info(i, energy_cut=energy_cut)

if __name__ == "__main__":
    main()