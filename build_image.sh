#!/bin/bash
buildah build -t registry.cjbal.com/mavis:latest --layers
buildah push registry.cjbal.com/mavis:latest