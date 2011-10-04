	.file	"asm_enc_offsets.c"
	.text
	.p2align 4,,15
.globl main
	.type	main, @function
main:
.LFB615:
	.cfi_startproc
#APP
# 23 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_coeff EQU $8
# 0 "" 2
# 24 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_zbin EQU $40
# 0 "" 2
# 25 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_round EQU $56
# 0 "" 2
# 26 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_quant EQU $16
# 0 "" 2
# 27 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_quant_fast EQU $24
# 0 "" 2
# 28 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_zbin_extra EQU $64
# 0 "" 2
# 29 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_zrun_zbin_boost EQU $48
# 0 "" 2
# 30 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_quant_shift EQU $32
# 0 "" 2
# 32 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_qcoeff_base EQU $0
# 0 "" 2
# 33 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_qcoeff_offset EQU $8
# 0 "" 2
# 34 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_dequant EQU $64
# 0 "" 2
# 35 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_dqcoeff_base EQU $16
# 0 "" 2
# 36 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_dqcoeff_offset EQU $24
# 0 "" 2
# 37 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_eob EQU $168
# 0 "" 2
# 40 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_base_src EQU $72
# 0 "" 2
# 41 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_src EQU $80
# 0 "" 2
# 42 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_src_diff EQU $0
# 0 "" 2
# 43 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_block_src_stride EQU $84
# 0 "" 2
# 45 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_predictor_base EQU $32
# 0 "" 2
# 46 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_blockd_predictor_offset EQU $40
# 0 "" 2
# 49 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_lowvalue EQU $0
# 0 "" 2
# 50 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_range EQU $4
# 0 "" 2
# 51 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_value EQU $8
# 0 "" 2
# 52 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_count EQU $12
# 0 "" 2
# 53 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_pos EQU $16
# 0 "" 2
# 54 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_writer_buffer EQU $24
# 0 "" 2
# 56 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenextra_token EQU $10
# 0 "" 2
# 57 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenextra_extra EQU $8
# 0 "" 2
# 58 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenextra_context_tree EQU $0
# 0 "" 2
# 59 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenextra_skip_eob_node EQU $11
# 0 "" 2
# 60 "vp8/encoder/asm_enc_offsets.c" 1
	
TOKENEXTRA_SZ EQU $16
# 0 "" 2
# 62 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_extra_bit_struct_sz EQU $24
# 0 "" 2
# 64 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_token_value EQU $0
# 0 "" 2
# 65 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_token_len EQU $4
# 0 "" 2
# 67 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_extra_bit_struct_tree EQU $0
# 0 "" 2
# 68 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_extra_bit_struct_prob EQU $8
# 0 "" 2
# 69 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_extra_bit_struct_len EQU $16
# 0 "" 2
# 70 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_extra_bit_struct_base_val EQU $20
# 0 "" 2
# 72 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_comp_tplist EQU $186048
# 0 "" 2
# 73 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_comp_common EQU $109536
# 0 "" 2
# 74 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_comp_bc2 EQU $133104
# 0 "" 2
# 76 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenlist_start EQU $0
# 0 "" 2
# 77 "vp8/encoder/asm_enc_offsets.c" 1
	
tokenlist_stop EQU $8
# 0 "" 2
# 78 "vp8/encoder/asm_enc_offsets.c" 1
	
TOKENLIST_SZ EQU $16
# 0 "" 2
# 80 "vp8/encoder/asm_enc_offsets.c" 1
	
vp8_common_mb_rows EQU $13252
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	ret
	.cfi_endproc
.LFE615:
	.size	main, .-main
	.ident	"GCC: (Ubuntu/Linaro 4.5.2-8ubuntu4) 4.5.2"
	.section	.note.GNU-stack,"",@progbits
