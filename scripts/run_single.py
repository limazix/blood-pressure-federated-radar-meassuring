#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click


@click.command()
@click.option('--data-dir')
@click.option('--subject-id')
@click.option('--port')
def main(data_dir, subject_id, port):
    print(data_dir, subject_id, port)


if __name__ == "__main__":
    main()
