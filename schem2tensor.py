"""
The script parses .schem files into tensors. The output has the following structure

output_dir/
    structure_1/
        block_type.pt
        attribute_pair_1/
            values.pt
            mask.pt
        attribute_pair_2/
            values.pt
            mask.pt

        ...

        attribute_pair_N/
            values.pt
            mask.pt
    ...
"""


import argparse
import io
import json
import os
from pathlib import Path

import nbtlib
import numpy as np
import torch
from immutable_views import *
from nbtlib.tag import *
from tqdm import tqdm

from src.utils.model_utils import AIR, AIR_BLOCK_IDX, BLOCK_TYPE, INFESTED, get_head_key


class _VarintIO:
    """"""

    ### --- Fields

    # Used for reading the int bits from a varint byte
    _INT_BITMASK = 0b0111_1111
    # Used for reading the continue bit in a byte, which is
    # the MSB, if one, we continue reading to the next byte.
    _CONTINUE_BITMASK = 0b1000_0000

    ### ---

    ### --- Public static methods

    """
    Reads and returns the next varint of a varint byte stream while making it advance.
    Note that the varint read is going to be positive I was too lazy
    """

    @staticmethod
    def readPositiveVarInt(stream: io.BytesIO) -> int:
        # Setup
        varint: int = 0
        positionInInt: int = 0

        # Going while we didn't finish reading the int
        while True:
            # Read the next byte of the stream
            currentByte = int.from_bytes(stream.read(1), "big", signed=False)
            # Add the int bits to our varint
            varint |= (currentByte & _VarintIO._INT_BITMASK) << positionInInt

            # If the continue bit is 0, then stop reading this int
            if (currentByte & _VarintIO._CONTINUE_BITMASK) == 0:
                break

            # If we are continuing, add 7 to the position in the int, since
            # in varints the int is separated in groups of 7 bits.
            positionInInt += 7

        return varint


def _initFromFile(schematicToLoadPath: str):
    # Get the schematic file as a nbt map
    schematicFile = nbtlib.load(schematicToLoadPath)
    fileBase = (
        schematicFile["Schematic"] if "Schematic" in schematicFile else schematicFile
    )

    ## Init the block palette
    filePalette = fileBase["Palette"]
    structureBlockPalette = {}

    for blockState, idTagInPalette in filePalette.items():
        idInPalette = int(idTagInPalette)
        structureBlockPalette[blockState] = idInPalette
        structureBlockPalette[idInPalette] = blockState
    # Nothing has been put inside the block palette, so we default init it.
    if len(structureBlockPalette) == 0:
        structureBlockPalette = {"minecraft:air": 0, 0: "minecraft:air"}
    # Set the free Id to the length of the block palette // 2 as each Id has 2 entries
    structureBlockPaletteFreeId = len(structureBlockPalette) // 2

    # -- Re process the block palette so that ID 0 is air
    # Put air inside the hashmap if it wasn't present yet.
    # used for future processing that's why it's not equal to 0
    if "minecraft:air" not in structureBlockPalette:
        structureBlockPalette["minecraft:air"] = structureBlockPaletteFreeId
        structureBlockPalette[structureBlockPaletteFreeId] = "minecraft:air"
        structureBlockPaletteFreeId += 1
    # If the current air ID isn't 0, switch it up with the
    # current 0 id, example:
    #   palette{0: black_wool, 1: air} will get switched to:
    #   palette{0: air, 1: black_wool}
    # And the byte ids will get processed afterwards
    beforeProcessingAirId = structureBlockPalette["minecraft:air"]
    airOldId = beforeProcessingAirId
    if beforeProcessingAirId != 0:
        beforeProcessingId0State = structureBlockPalette[0]
        # Pop the 0 id
        structureBlockPalette.pop(0)
        structureBlockPalette.pop(beforeProcessingId0State)
        # Pop the air id
        structureBlockPalette.pop(beforeProcessingAirId)
        structureBlockPalette.pop("minecraft:air")

        # Put the air at 0
        structureBlockPalette[0] = "minecraft:air"
        structureBlockPalette["minecraft:air"] = 0
        # Put the old currentId0State where air was
        structureBlockPalette[beforeProcessingAirId] = beforeProcessingId0State
        structureBlockPalette[beforeProcessingId0State] = beforeProcessingAirId

    ## Init the blockStates in _blockStates
    structureBlockStates: dict[tuple[int, int, int], int] = {}
    if "BlockData" in fileBase:
        # Get the necessary data for blockState loading
        fileBlockStatesIds = fileBase["BlockData"]
        blockStatesIds = bytearray(fileBlockStatesIds)
        schemOffset = (0, 0, 0)  # fileBase['Offset']
        # schemHeight = int(fileBase["Height"])  # y
        schemLength = int(fileBase["Length"])  # z
        schemWidth = int(fileBase["Width"])  # x

        # Variable so that we don't have to do a multiplication every iteration
        schemYSliceArea = schemWidth * schemLength

        if len(structureBlockPalette) < 128:
            # The amount of block states is less than 128, so each block is a byte,
            # so we use the old faster algorithm to load the structure

            for blockStateIndex, blockStateId in enumerate(blockStatesIds):
                # Process the blockStateId since the palette has been modified
                processedBlockStateId = blockStateId
                # -- Do le switcharoo, air new Id is 0 no matter what
                # If we refer to the old air Id, it means that we refer to air now
                # which is 0
                if blockStateId == airOldId:
                    processedBlockStateId = 0
                # If we refer to the old id at 0, we are referring to the new
                # content of the old air id spot
                if blockStateId == 0:
                    processedBlockStateId = airOldId

                # Since we processed the blockPalette so that air is 0, if the ID is 0
                # then skip, as we don't keep track of air blocks

                # Getting the coordinates in the schem not shifted yet
                blockStateSchemY = blockStateIndex // schemYSliceArea
                blockStateSchemZ = (blockStateIndex % schemYSliceArea) // schemWidth
                blockStateSchemX = blockStateIndex % schemWidth

                # Shift the coordinates so that the blocks are back in their right position
                realY = blockStateSchemY + schemOffset[1]
                realZ = blockStateSchemZ + schemOffset[2]
                realX = blockStateSchemX + schemOffset[0]

                # Place the block
                structureBlockStates[(realX, realY, realZ)] = processedBlockStateId

        else:
            # The palette contains more (or equal) than 128 entries, so we use
            # the varint algorithm

            # Put the blockState bytes into a BytesIO for a stream
            blockStatesIdStream = io.BytesIO(blockStatesIds)

            # Setup before loopin woo
            blockStateIndex = 0

            # We loopin bois
            while blockStatesIdStream.tell() < len(blockStatesIds):
                # Get the next varint of the stream
                blockStateId = _VarintIO.readPositiveVarInt(blockStatesIdStream)
                # From there, use the normal old algorithm, cba to put it in
                # a method lmfao

                # ===
                # Process the blockStateId since the palette has been modified
                processedBlockStateId = blockStateId
                # -- Do le switcharoo, air new Id is 0 no matter what
                # If we refer to the old air Id, it means that we refer to air now
                # which is 0
                if blockStateId == airOldId:
                    processedBlockStateId = 0
                # If we refer to the old id at 0, we are referring to the new
                # content of the old air id spot
                if blockStateId == 0:
                    processedBlockStateId = airOldId

                # Since we processed the blockPalette so that air is 0, if the ID is 0
                # then skip, as we don't keep track of air blocks

                # Getting the coordinates in the schem not shifted yet
                blockStateSchemY = blockStateIndex // schemYSliceArea
                blockStateSchemZ = (blockStateIndex % schemYSliceArea) // schemWidth
                blockStateSchemX = blockStateIndex % schemWidth

                # Shift the coordinates so that the blocks are back in their right position
                realY = blockStateSchemY + schemOffset[1]
                realZ = blockStateSchemZ + schemOffset[2]
                realX = blockStateSchemX + schemOffset[0]

                # Place the block
                structureBlockStates[(realX, realY, realZ)] = processedBlockStateId
                # ===

                # -- Increment block state index as we're done registering that block
                blockStateIndex += 1

    return structureBlockStates, structureBlockPalette


def create_parser():
    parser = argparse.ArgumentParser(description="Generate world dataset")

    parser.add_argument("--schem-dir", required=True, type=str)

    parser.add_argument("--output-dir", required=False, type=str)

    parser.add_argument(
        "--block-data-dir", required=False, type=str, default="./src/block_data"
    )

    parser.add_argument(
        "--n-workers",
        required=False,
        type=int,
    )

    parser.add_argument(
        "--limit",
        required=False,
        type=int,
    )
    return parser


def parse_block(block: str):
    """
    block : str
        A string representing minecraft block with its attributes.
        Example: minecraft:water[level=0]
    """
    idx = block.find("[")
    if idx != -1:
        block, attr_data = block[:idx], block[idx + 1 : -1]
    else:
        return block, {}

    attr_data = attr_data.split(",")
    attr_dict = {}
    for item in attr_data:
        attr, value = item.split("=")
        attr_dict[attr] = value
    return block, attr_dict


def filter_attribute_dict(
    block: str,
    attr_dict: dict[str, str],
    attributes_defaults: dict[str, str],
    block_attributes_defaults: dict[str, dict[str, str]],
    filtered_blocks_dict: dict[str, dict[str, list[str]]],
) -> dict[str, int]:
    """
    block : str
        A string representing the type of block
    attr_dict : dict
        A dict representing the attribute data of this block
    attributes_defaults : dict
        A dict of default attributes for all blocks
    block_attributes_defaults : dict
        A dict of default attributes for specific blocks
    filtered_blocks_dict : dict
        A dict containg an attribute dict with the values spectre for each block

    Returns : dict[str, int]
        A filtered attr dict without defaults and with indexes
    """
    final_attr_dict = {}
    for attr in attr_dict:
        if attr not in attributes_defaults:
            if block in block_attributes_defaults:
                if attr not in block_attributes_defaults[block]:
                    final_attr_dict[attr] = attr_dict[attr]
            else:
                final_attr_dict[attr] = attr_dict[attr]

    for attr in final_attr_dict:
        value = final_attr_dict[attr]
        block_attrs = filtered_blocks_dict[block]
        final_attr_dict[attr] = sorted(block_attrs[attr]).index(value)

    return final_attr_dict


def block_to_idx(block: str, block2idx: dict[str, int]) -> int:
    """
    A function that matches the block type and its index,
    considering filtered blocks. Infested blocks are treated
    as their normal variant. Other filtered blocks are treated
    as air blocks.

    Args:

    block : str
        A string representing minecraft block type (ex. 'minecraft:stone')
    block2idx : dict
        A dict for matching block type and its index
    """

    if block not in block2idx:
        if block.startswith(INFESTED):
            block = "minecraft:" + block[len(INFESTED) :]
        else:
            block = AIR
    return block, block2idx[block]


def create_block2idx_mapping(block_data_dir):
    with open(block_data_dir / "filtered_blocks.json") as f:
        filtered_blocks_dict = json.load(f)
    idx2block = sorted(list(filtered_blocks_dict.keys()))
    air_idx = idx2block.index(AIR)
    idx2block[AIR_BLOCK_IDX], idx2block[air_idx] = (
        idx2block[air_idx],
        idx2block[AIR_BLOCK_IDX],
    )
    block2idx = {idx2block[i]: i for i in range(len(idx2block))}
    with open(block_data_dir / "idx2block.json", "w") as w:
        json.dump(idx2block, w, indent=4)
    with open(block_data_dir / "block2idx.json", "w") as w:
        json.dump(block2idx, w, indent=4)


def parse_schematics(
    data_dir, output_dir=None, block_data_dir="src/block_data", limit=None
):
    """
    A function for turning schematics in folder into tensors
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / "parsed"
    output_dir = Path(output_dir)

    block_data_dir = Path(block_data_dir)

    with open(block_data_dir / "filtered_blocks.json") as f:
        filtered_blocks_dict = json.load(f)

    if not (
        os.path.exists(block_data_dir / "idx2block.json")
        and os.path.exists(block_data_dir / "block2idx.json")
    ):
        create_block2idx_mapping(block_data_dir)

    with open(block_data_dir / "block2idx.json") as f:
        block2idx = json.load(f)

    with open(block_data_dir / "attributes_defaults.json") as f:
        attributes_defaults = json.load(f)
    with open(block_data_dir / "block_attributes_defaults.json") as f:
        block_attributes_defaults = json.load(f)

    with open(block_data_dir / "block_attributes_defaults.json") as f:
        block_attributes_defaults = json.load(f)

    with open(block_data_dir / "non_default_attribute_pairs.json") as f:
        non_default_attribute_pairs = json.load(f)

    with open(block_data_dir / "attr_pair2idxs.json") as f:
        attr_pair2idxs = json.load(f)

    for key in attr_pair2idxs:
        attr_pair2idxs[key] = torch.tensor(attr_pair2idxs[key], dtype=torch.int16)

    files = sorted(os.listdir(data_dir))

    if limit is None:
        limit = len(files)

    for i, schm in enumerate(tqdm(files, total=limit)):
        if i >= limit:
            break
        file: Path = data_dir / schm
        if file.suffix == ".schem":
            structure_name = file.stem
            try:
                schem = nbtlib.load(file)
            except Exception as e:
                print(f"Error: {e}\nFilename: {schm}")
                continue
            length, width, height = schem["Length"], schem["Width"], schem["Height"]
            del schem

            coord2byte, palette = _initFromFile(file)

            block_grid_tensor = torch.zeros(
                (width, height, length), dtype=torch.int16
            )  # x, y, z

            attributes = {}

            for x, y, z in coord2byte:
                block_byte = coord2byte[(x, y, z)]
                block = palette[block_byte]

                block, attr_dict = parse_block(block)  # str, dict
                block, block_idx = block_to_idx(block, block2idx)  # str, int
                if block == AIR:
                    attr_dict = {}

                attr_dict = filter_attribute_dict(
                    block=block,
                    attr_dict=attr_dict,
                    attributes_defaults=attributes_defaults,
                    block_attributes_defaults=block_attributes_defaults,
                    filtered_blocks_dict=filtered_blocks_dict,
                )

                block_grid_tensor[x][y][z] = block_idx
                if len(attr_dict):
                    attributes[(x, y, z)] = attr_dict

            os.makedirs(output_dir / structure_name, exist_ok=True)
            torch.save(
                block_grid_tensor, output_dir / structure_name / f"{BLOCK_TYPE}.pt"
            )  # int16

            # create masks and attr vectors for each attr-value pair
            for attr, values in non_default_attribute_pairs:
                head_key = get_head_key(attr, values)
                mask = torch.isin(block_grid_tensor, attr_pair2idxs[head_key])
                idxs = torch.nonzero(mask)

                attribute_values = []
                for x, y, z in idxs:
                    attribute_values.append(
                        attributes[(x.item(), y.item(), z.item())][attr]
                    )

                os.makedirs(output_dir / structure_name / head_key, exist_ok=True)

                torch.save(
                    mask, output_dir / structure_name / head_key / "mask.pt"
                )  # bool
                torch.save(
                    torch.tensor(attribute_values, dtype=torch.int8),
                    output_dir / structure_name / head_key / "values.pt",
                )  # int8


def main(args):
    data_dir = args.schem_dir
    output_dir = args.output_dir
    block_data_dir = args.block_data_dir
    limit = args.limit
    parse_schematics(data_dir, output_dir, block_data_dir, limit)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
