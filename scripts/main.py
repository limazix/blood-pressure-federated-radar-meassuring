#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click


@click.command()
def run():
    """Method used to run the application from cli"""
    print("Run!!!")


if __name__ == "__main__":
    run()
